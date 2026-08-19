"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

import json
import logging
import time
import traceback
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model, __version__
from minisweagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded, TimeExceeded
from minisweagent.utils.serialize import recursive_merge


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    wall_time_limit_seconds: int = 0
    """Stop agent after this many seconds of wall-clock time. 0 means no limit."""
    max_consecutive_format_errors: int = 3
    """Exit after this many format errors in a row (0 = no limit)."""
    output_path: Path | None = None
    """Save the trajectory to this path."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        self._start_time = time.time()

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "elapsed_seconds": int(time.time() - self._start_time),
            },
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        # 保存本次任务和额外变量，供 YAML 模板中的 {{ task }} 等占位符使用。
        self.extra_template_vars |= {"task": task, **kwargs}
        # 每次 run 都从一段全新的对话轨迹开始。
        self.messages = []
        self.add_messages(
            # 第一条：渲染 system_template，规定角色、输出格式和不可违反的规则。
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            # 第二条：渲染 instance_template，注入当前任务、推荐工作流和环境信息。
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        # 这两个初始提示只在这里创建一次；随后每轮都会复用并继续追加同一个 self.messages。
        while True:
            # Agent Loop 的反复执行入口：每一轮尝试完成“模型决策 → 工具执行 → 结果回填”。
            try:
                # step() 内部先调用模型，再执行模型请求的 actions；具体流程见下方 step()。
                self.step()
                # 本轮没有格式错误，连续格式错误次数归零。
                self.n_consecutive_format_errors = 0  # reset on any clean step
            except FormatError as e:
                # 模型输出无法解析成预期的 action/tool call，例如没有工具调用、缺少 command 参数。
                # 将纠错提示加入上下文，让模型下一轮按约定格式重试；连续失败太多才退出。
                # The call was billed before parsing failed, so query() never got to charge it.
                # 请求已经产生模型费用，但 query() 在解析失败时未能走到 self.cost += ...，所以这里补记费用。
                self.cost += e.messages[0].get("extra", {}).get("cost", 0.0)
                self.n_consecutive_format_errors += 1
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                # 一个“可预期的流程中断”信号，不只代表 Ctrl+C：也可能是完成、超预算、超时或用户补充任务。
                # 是否退出取决于 e.messages 中最后一条是否带 role="exit"，不是由异常父类本身决定。
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                # 无论本轮成功或异常，都保存当前完整轨迹：messages、费用/调用次数、模型与环境状态、最终状态等。
                # 用于中断后排查、复盘和重放；不是发给模型的额外上下文。
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                # role="exit" 是本项目内部的终止哨兵：正常完成、超限或未捕获异常都会追加它，然后退出循环。
                break
        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        # 一个完整回合：query() 只负责调用模型并追加其回复；execute_actions() 再执行回复中的 actions，
        # 并把工具输出追加回 messages，供下一回合模型读取。
        # query() 是实际调用一次 LLM；execute_actions() 则执行已由模型适配层解析好的 tool actions。
        return self.execute_actions(self.query())

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            # 不是并发上限：step_limit 是本次 run 已调用模型的次数上限；cost_limit 是累计模型费用上限。
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            # 不是单次 LLM 调用超时：这是从本次 run 开始到现在的总墙钟时间（wall-clock time）上限。
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        # 每经过一轮 query() 才算一次模型调用；一个用户任务通常会经历多轮，而非只调用一次模型。
        self.n_calls += 1
        # 发送的是当前完整轨迹：初始 system/user 提示 + 此前的模型回复 + 工具执行结果。
        # 因此不是每一轮都重新插入一份 instance_template。
        # 下面三行：发起一次模型调用；将本次回复（已归一化，含 actions/cost 等 extra）追加到 messages 尾部；累计其费用。
        message = self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        # message["extra"]["actions"] 是模型回复中解析出的工具请求；self.env.execute(action) 才是真正执行工具的地方。
        # 这里的 Environment 是本地 Bash 环境，因此 action 通常就是待执行的 command。
        outputs = [self.env.execute(action) for action in message.get("extra", {}).get("actions", [])]
        # 将工具输出格式化成下一轮模型可读的 observation/tool-result 消息，并追加到对话轨迹。
        # 上面工具执行的结果，再拼到messages里面
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
