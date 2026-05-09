# SGLang 模型推理流程

本文面向想阅读 SGLang Runtime, 简称 SRT, 推理主链路的开发者。范围限定在文本生成模型的在线推理路径, 即 `/generate`、OpenAI 兼容接口和 Python `Engine.generate()` 最终进入的同一条 SRT 生成链路。embedding、reward、disaggregation、speculative decoding、pipeline parallel 等特性会在对应位置点到为止, 主线仍以普通 causal LM 生成为准。

文中的代码块都标出具体文件, 但为了突出主路径, 部分片段会用 `...` 省略错误处理、指标、分布式分支或旁路逻辑。

## 一句话总览

SGLang 的默认推理服务由主进程和两个子进程组成:

1. HTTP/Engine/TokenizerManager 在主进程中接收请求、套模板、tokenize, 并把 `TokenizedGenerateReqInput` 通过 ZMQ 发给 Scheduler。
2. Scheduler 子进程维护等待队列、运行队列、prefix cache 和 KV cache, 将请求组织成 `ScheduleBatch -> ModelWorkerBatch -> ForwardBatch`, 调用模型 forward 并采样下一 token。
3. DetokenizerManager 子进程把 Scheduler 输出的 token id 增量 decode 成文本, 再发回 TokenizerManager, 最后由 HTTP/OpenAI/Python API 返回给用户。

代码里对这个结构有直接说明:

```python
# python/sglang/srt/entrypoints/http_server.py
def launch_server(...):
    """
    Launch SRT (SGLang Runtime) Server.

    The SRT server consists of an HTTP server and an SRT engine.

    - HTTP server: A FastAPI server that routes requests to the engine.
    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager,
           schedules batches, forwards them, and sends the output tokens to the
           Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and
           sends the result back to the Tokenizer Manager.
    """
```

同样的描述也在 Python API 入口处:

```python
# python/sglang/srt/entrypoints/engine.py
class Engine(EngineScoreMixin, EngineBase):
    """
    The entry point to the inference engine.

    - The engine consists of three components:
        1. TokenizerManager: Tokenizes the requests and sends them to the scheduler.
        2. Scheduler (subprocess): Receives requests from the Tokenizer Manager,
           schedules batches, forwards them, and sends the output tokens to the
           Detokenizer Manager.
        3. DetokenizerManager (subprocess): Detokenizes the output tokens and
           sends the result back to the Tokenizer Manager.
    """
```

## 关键文件索引

| 阶段 | 文件 | 作用 |
| --- | --- | --- |
| 服务入口 | `python/sglang/launch_server.py` | 选择 HTTP、gRPC、Ray 或 encoder-only 启动路径 |
| HTTP 路由 | `python/sglang/srt/entrypoints/http_server.py` | FastAPI 路由, `/generate` 和 `/v1/chat/completions` 都在这里注册 |
| OpenAI 适配 | `python/sglang/srt/entrypoints/openai/serving_chat.py`, `serving_completions.py`, `serving_base.py` | 将 OpenAI 请求转换成内部 `GenerateReqInput` |
| Python API | `python/sglang/srt/entrypoints/engine.py` | `Engine.generate()` 构造 `GenerateReqInput` 并复用 TokenizerManager |
| 请求结构 | `python/sglang/srt/managers/io_struct.py` | `GenerateReqInput`, `TokenizedGenerateReqInput`, 各类输出结构 |
| Tokenizer | `python/sglang/srt/managers/tokenizer_manager.py` | normalize, tokenize, 发送 scheduler, 接收 detokenizer 输出 |
| Scheduler | `python/sglang/srt/managers/scheduler.py` | 请求入队、动态 batching、prefill/decode 调度、forward 调用 |
| Batch 状态 | `python/sglang/srt/managers/schedule_batch.py` | `Req`, `ScheduleBatch`, `ModelWorkerBatch`, batch tensor 准备 |
| Forward 状态 | `python/sglang/srt/model_executor/forward_batch_info.py` | `ForwardMode`, `ForwardBatch` |
| TP Worker | `python/sglang/srt/managers/tp_worker.py` | 将 `ModelWorkerBatch` 转为 `ForwardBatch`, 调用 `ModelRunner` |
| ModelRunner | `python/sglang/srt/model_executor/model_runner.py` | 加载模型、管理 KV pool、执行 prefill/decode forward 和 sample |
| 模型实现示例 | `python/sglang/srt/models/qwen2.py` | 模型层 forward、lm head、logits processor |
| 采样 | `python/sglang/srt/sampling/sampling_params.py`, `sampling_batch_info.py`, `python/sglang/srt/layers/sampler.py` | 采样参数、batch 化采样元数据、greedy/top-k/top-p/min-p 采样 |
| 输出处理 | `python/sglang/srt/managers/scheduler_output_processor_mixin.py` | 更新请求状态、判断完成、发送 token id 输出 |
| Detokenizer | `python/sglang/srt/managers/detokenizer_manager.py` | token id 到 text 的增量 detokenize |

## 1. 服务启动和进程结构

命令行入口在 `python/sglang/launch_server.py`。默认路径会进入 HTTP server:

```python
# python/sglang/launch_server.py
def run_server(server_args):
    """Run the server based on server_args.grpc_mode and server_args.encoder_only."""
    if server_args.encoder_only:
        ...
    elif server_args.grpc_mode:
        ...
    elif server_args.use_ray:
        ...
    else:
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)
```

HTTP server 启动时通过 `Engine._launch_subprocesses()` 拉起 Scheduler 和 Detokenizer:

```python
# python/sglang/srt/entrypoints/http_server.py
(
    tokenizer_manager,
    template_manager,
    port_args,
    scheduler_init_result,
    subprocess_watchdog,
) = Engine._launch_subprocesses(
    server_args=server_args,
    init_tokenizer_manager_func=init_tokenizer_manager_func,
    run_scheduler_process_func=run_scheduler_process_func,
    run_detokenizer_process_func=run_detokenizer_process_func,
)
```

`Engine._launch_subprocesses()` 做三件关键事:

```python
# python/sglang/srt/entrypoints/engine.py
# Launch scheduler processes
scheduler_init_result, scheduler_procs = cls._launch_scheduler_processes(
    server_args, port_args, run_scheduler_process_func
)

# Launch detokenizer process
detoken_proc = mp.Process(
    target=run_detokenizer_process_func,
    args=(
        server_args,
        port_args,
    ),
)
detoken_proc.start()

# Init tokenizer manager first, as the bootstrap server is initialized here
if server_args.tokenizer_worker_num == 1:
    tokenizer_manager, template_manager = init_tokenizer_manager_func(
        server_args, port_args
    )
```

Scheduler 按 TP/PP rank 起多个子进程。普通单 DP 情况下, 每个 `pp_rank x tp_rank` 一个 scheduler process:

```python
# python/sglang/srt/entrypoints/engine.py
for pp_rank in pp_rank_range:
    for tp_rank in tp_rank_range:
        reader, writer = mp.Pipe(duplex=False)
        ...
        proc = mp.Process(
            target=run_scheduler_process_func,
            args=(
                server_args,
                port_args,
                gpu_id,
                tp_rank,
                attn_cp_rank,
                moe_dp_rank,
                moe_ep_rank,
                pp_rank,
                None,
                writer,
            ),
        )
        proc.start()
```

Scheduler 子进程入口是 `run_scheduler_process()`:

```python
# python/sglang/srt/managers/scheduler.py
def run_scheduler_process(...):
    ...
    scheduler = Scheduler(
        server_args,
        port_args,
        gpu_id,
        tp_rank,
        moe_ep_rank,
        pp_rank,
        attn_cp_rank,
        moe_dp_rank,
        dp_rank,
    )

    # Send initialization info back to the parent process
    pipe_writer.send(scheduler.get_init_info())

    # Run the event loop (blocks until shutdown)
    scheduler.run_event_loop()
```

## 2. 请求入口: `/generate`, OpenAI API, Python API

### Native `/generate`

FastAPI 的 `/generate` 路由直接接收 `GenerateReqInput`, 然后调用 `TokenizerManager.generate_request()`:

```python
# python/sglang/srt/entrypoints/http_server.py
@app.api_route(
    "/generate",
    methods=["POST", "PUT"],
    response_class=SGLangORJSONResponse,
)
async def generate_request(obj: GenerateReqInput, request: Request):
    """Handle a generate request."""
    if obj.stream:
        async def stream_results() -> AsyncIterator[bytes]:
            async for out in _global_state.tokenizer_manager.generate_request(
                obj, request
            ):
                yield b"data: " + dumps_json(out) + b"\n\n"
            yield b"data: [DONE]\n\n"

        return StreamingResponse(...)
    else:
        ret = await _global_state.tokenizer_manager.generate_request(
            obj, request
        ).__anext__()
        return orjson_response(ret)
```

### OpenAI `/v1/chat/completions`

OpenAI 路由先进入对应 serving 类:

```python
# python/sglang/srt/entrypoints/http_server.py
@app.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
async def openai_v1_chat_completions(
    request: ChatCompletionRequest, raw_request: Request
):
    """OpenAI-compatible chat completion endpoint."""
    return await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )
```

OpenAI 通用基类会做 validate, 转内部请求, 然后按 stream/non-stream 处理:

```python
# python/sglang/srt/entrypoints/openai/serving_base.py
async def handle_request(self, request: OpenAIServingRequest, raw_request: Request):
    # Validate request
    error_msg = self._validate_request(request)
    ...

    # Convert to internal format
    adapted_request, processed_request = self._convert_to_internal_request(
        request, raw_request
    )

    if hasattr(request, "stream") and request.stream:
        return await self._handle_streaming_request(
            adapted_request, processed_request, raw_request
        )
    else:
        return await self._handle_non_streaming_request(
            adapted_request, processed_request, raw_request
        )
```

Chat completion 的转换核心是: 处理 messages 和 chat template, 构造 `sampling_params`, 最后生成内部 `GenerateReqInput`:

```python
# python/sglang/srt/entrypoints/openai/serving_chat.py
def _convert_to_internal_request(
    self,
    request: ChatCompletionRequest,
    raw_request: Request = None,
) -> tuple[GenerateReqInput, ChatCompletionRequest]:
    # Process messages and apply chat template
    processed_messages = self._process_messages(request, is_multimodal)

    # Build sampling parameters
    sampling_params = request.to_sampling_params(
        stop=processed_messages.stop,
        model_generation_config=self.default_sampling_params,
        tool_call_constraint=processed_messages.tool_call_constraint,
    )

    adapted_request = GenerateReqInput(
        **prompt_kwargs,
        image_data=processed_messages.image_data,
        video_data=processed_messages.video_data,
        audio_data=processed_messages.audio_data,
        sampling_params=sampling_params,
        return_logprob=request.logprobs,
        stream=request.stream,
        lora_path=lora_path,
        routed_dp_rank=effective_routed_dp_rank,
        rid=request.rid,
        extra_key=self._compute_extra_key(request),
        priority=request.priority,
        custom_logit_processor=request.custom_logit_processor,
        ...
    )

    return adapted_request, request
```

非 streaming 的 chat completion 最终还是等 `TokenizerManager.generate_request()`:

```python
# python/sglang/srt/entrypoints/openai/serving_chat.py
async def _handle_non_streaming_request(
    self,
    adapted_request: GenerateReqInput,
    request: ChatCompletionRequest,
    raw_request: Request,
):
    ret = await self.tokenizer_manager.generate_request(
        adapted_request, raw_request
    ).__anext__()
    ...
    response = self._build_chat_response(request, ret, int(time.time()))
    return response
```

### Python `Engine.generate()`

离线 Python API 入口也构造同一个 `GenerateReqInput`, 然后复用 TokenizerManager:

```python
# python/sglang/srt/entrypoints/engine.py
def generate(...):
    obj = GenerateReqInput(
        text=prompt,
        input_ids=input_ids,
        sampling_params=sampling_params,
        image_data=image_data,
        audio_data=audio_data,
        video_data=video_data,
        return_logprob=return_logprob,
        stream=stream,
        lora_path=lora_path,
        rid=rid,
        priority=priority,
    )
    generator = self.tokenizer_manager.generate_request(obj, None)

    if stream:
        ...
    else:
        ret = self.loop.run_until_complete(generator.__anext__())
        return ret
```

## 3. 请求数据结构: `GenerateReqInput` 到 `TokenizedGenerateReqInput`

外部请求最初会落到 `GenerateReqInput`:

```python
# python/sglang/srt/managers/io_struct.py
@dataclass
class GenerateReqInput(BaseReq):
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[str], str]] = None
    # The token ids for text; one can specify either text or input_ids
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # The sampling_params. See descriptions below.
    sampling_params: Optional[Union[List[Dict], Dict]] = None
    # Whether to stream output.
    stream: bool = False
    # Session info for continual prompting
    session_params: Optional[Union[List[Dict], Dict]] = None
    # The path to the LoRA adaptors
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None
```

`GenerateReqInput.normalize_batch_and_arguments()` 会校验输入, 判断单条还是 batch, 处理 parallel sampling:

```python
# python/sglang/srt/managers/io_struct.py
def normalize_batch_and_arguments(self):
    """
    Normalize the batch size and arguments for the request.
    """
    self._validate_inputs()
    self._determine_batch_size()
    self._handle_parallel_sampling()

    if self.is_single:
        self._normalize_single_inputs()
    else:
        self._normalize_batch_inputs()

    self._validate_rid_uniqueness()
```

tokenize 后会变成 `TokenizedGenerateReqInput`, 它是 Scheduler 接收的生成请求:

```python
# python/sglang/srt/managers/io_struct.py
@dataclass
class TokenizedGenerateReqInput(BaseReq):
    # The input text
    input_text: str
    # The input token ids
    input_ids: List[int]
    # The multimodal inputs
    mm_inputs: object
    # The sampling parameters
    sampling_params: SamplingParams
    # Whether to return the logprobs
    return_logprob: bool
    # Whether to stream output
    stream: bool
    ...
```

## 4. TokenizerManager: tokenize, 发给 Scheduler, 等输出

TokenizerManager 初始化时会读取模型配置、初始化 tokenizer/multimodal processor 和 IPC socket:

```python
# python/sglang/srt/managers/tokenizer_manager.py
class TokenizerManager(TokenizerControlMixin, TokenizerManagerScoreMixin):
    """TokenizerManager is a process that tokenizes the text."""

    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        self.server_args = server_args
        ...
        # Init model config
        self.init_model_config()

        # Initialize tokenizer and multimodalprocessor
        self.init_tokenizer_and_processor()

        # Init inter-process communication
        self.init_ipc_channels(port_args)
```

它的 IPC 方向是:

```python
# python/sglang/srt/managers/tokenizer_manager.py
def init_ipc_channels(self, port_args: PortArgs):
    context = zmq.asyncio.Context(2)
    self.recv_from_detokenizer = get_zmq_socket(
        context, zmq.PULL, port_args.tokenizer_ipc_name, True
    )
    self.send_to_scheduler = get_zmq_socket(
        context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
    )
```

`generate_request()` 是主逻辑:

```python
# python/sglang/srt/managers/tokenizer_manager.py
async def generate_request(
    self,
    obj: Union[GenerateReqInput, EmbeddingReqInput],
    request: Optional[fastapi.Request] = None,
):
    self.auto_create_handle_loop()

    # Normalize the request
    obj.normalize_batch_and_arguments()
    self._set_default_priority(obj)

    self._init_req_state(obj, request)
    ...

    async with self.model_update_lock.reader_lock:
        await self._validate_and_resolve_lora(obj)

        # Tokenize the request and send it to the scheduler
        if obj.is_single:
            tokenized_obj = await self._tokenize_one_request(obj)
            self._send_one_request(tokenized_obj)
            async for response in self._wait_one_response(obj, request):
                yield response
        else:
            async for response in self._handle_batch_request(obj, request):
                yield response
```

单请求 tokenize 会优先使用用户传入的 `input_ids`, 否则调用 tokenizer; 多模态请求还会走 multimodal processor:

```python
# python/sglang/srt/managers/tokenizer_manager.py
async def _tokenize_one_request(self, obj):
    """Tokenize one request."""
    input_embeds = None
    input_text = obj.text
    ...
    if obj.input_embeds is not None:
        input_embeds = obj.input_embeds
        input_ids = obj.input_ids
    elif obj.input_ids is not None:
        input_ids = obj.input_ids
    else:
        input_ids, token_type_ids = await self._tokenize_texts(
            input_text, is_cross_encoder_request
        )

    contains_mm_input = obj.contains_mm_input()
    should_run_mm_processor = self.mm_processor is not None and (
        contains_mm_input or is_mossvl
    )
```

采样参数在 tokenized object 创建时被解析、normalize 和 verify:

```python
# python/sglang/srt/managers/tokenizer_manager.py
def _create_tokenized_object(...):
    if self.preferred_sampling_params:
        sampling_kwargs = {**self.preferred_sampling_params, **obj.sampling_params}
    else:
        sampling_kwargs = obj.sampling_params
    sampling_params = self.sampling_params_class(**sampling_kwargs)
    sampling_params.normalize(self.tokenizer)
    sampling_params.verify(self.model_config.vocab_size)

    tokenized_obj = TokenizedGenerateReqInput(
        input_text,
        input_ids,
        mm_inputs,
        sampling_params,
        obj.return_logprob,
        obj.logprob_start_len,
        obj.top_logprobs_num,
        obj.token_ids_logprob,
        obj.stream,
        rid=obj.rid,
        ...
    )
```

发送给 Scheduler 很直接:

```python
# python/sglang/srt/managers/tokenizer_manager.py
def _send_one_request(self, tokenized_obj):
    tokenized_obj.time_stats.set_api_server_dispatch_time()
    tokenized_obj = wrap_shm_features(tokenized_obj)
    self.send_to_scheduler.send_pyobj(tokenized_obj)
    tokenized_obj.time_stats.set_api_server_dispatch_finish_time()
```

同时 TokenizerManager 会启动后台 `handle_loop()` 从 Detokenizer 接输出:

```python
# python/sglang/srt/managers/tokenizer_manager.py
async def handle_loop(self):
    """The event loop that handles requests"""
    while True:
        recv_obj = await self.recv_from_detokenizer.recv_pyobj()
        if isinstance(
            recv_obj,
            (BatchStrOutput, BatchEmbeddingOutput, BatchTokenIDOutput),
        ):
            await self._handle_batch_output(recv_obj)
        else:
            self._result_dispatcher(recv_obj)
```

请求侧的 `_wait_one_response()` 等待 `ReqState.event`, 拿到 `out_list` 后 yield 给 HTTP/OpenAI/Python API:

```python
# python/sglang/srt/managers/tokenizer_manager.py
async def _wait_one_response(self, obj, request=None):
    """Wait for the response of one request."""
    state = self.rid_to_state[obj.rid]
    is_stream = getattr(obj, "stream", False)
    while True:
        await asyncio.wait_for(state.event.wait(), timeout=_REQUEST_STATE_WAIT_TIMEOUT)

        out_list = state.out_list
        state.out_list = []
        finished = state.finished
        state.event.clear()
        ...
        if finished:
            yield out
            break

        if is_stream:
            yield out
```

## 5. Scheduler: 入队、调度 prefill/decode、调用模型

Scheduler 初始化时会设置调度参数、IPC、tokenizer、模型 worker、cache 等组件:

```python
# python/sglang/srt/managers/scheduler.py
class Scheduler(...):
    """A scheduler that manages a tensor parallel GPU worker."""

    def __init__(...):
        self.server_args = server_args
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.schedule_policy = server_args.schedule_policy
        self.enable_overlap = not server_args.disable_overlap_schedule and not use_mlx()
        ...
        # Init model configs
        self.init_model_config()

        # Init inter-process communication
        self.init_ipc_channels(port_args)

        # Init tokenizer
        self.init_tokenizer()
```

请求分发器把 `TokenizedGenerateReqInput` 路由到 `handle_generate_request()`:

```python
# python/sglang/srt/managers/scheduler.py
def init_request_dispatcher(self):
    self._request_dispatcher = TypeBasedDispatcher(
        [
            (TokenizedGenerateReqInput, self.handle_generate_request),
            (TokenizedEmbeddingReqInput, self.handle_embedding_request),
            (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),
            ...
        ]
    )
```

普通 scheduler loop 是:

```python
# python/sglang/srt/managers/scheduler.py
@DynamicGradMode()
def event_loop_normal(self):
    """A normal scheduler loop."""
    while True:
        # Receive requests
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)
        if self._engine_paused:
            continue

        # Get the next batch to run
        batch = self.get_next_batch_to_run()
        self.cur_batch = batch

        # Launch the current batch
        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
        else:
            self.on_idle()

        self.last_batch = batch
```

`process_input_requests()` 调用 dispatcher, 也就是将 tokenized 请求转换成 scheduler 内部的 `Req`:

```python
# python/sglang/srt/managers/scheduler.py
def process_input_requests(self, recv_reqs: List):
    for recv_req in recv_reqs:
        output = self._request_dispatcher(recv_req)
        if output is not None:
            if not isinstance(output, RpcReqOutput):
                self.send_to_tokenizer.send_output(output, recv_req)
```

`handle_generate_request()` 构造 `Req`, 处理多模态、长度、logprob、grammar, 最后入队:

```python
# python/sglang/srt/managers/scheduler.py
def handle_generate_request(self, recv_req: TokenizedGenerateReqInput):
    ...
    req = Req(
        recv_req.rid,
        recv_req.input_text,
        recv_req.input_ids,
        recv_req.sampling_params,
        return_logprob=recv_req.return_logprob,
        top_logprobs_num=recv_req.top_logprobs_num,
        token_ids_logprob=recv_req.token_ids_logprob,
        stream=recv_req.stream,
        lora_id=recv_req.lora_id,
        input_embeds=recv_req.input_embeds,
        custom_logit_processor=recv_req.custom_logit_processor,
        eos_token_ids=self.model_config.hf_eos_token_id,
        priority=recv_req.priority,
        time_stats=recv_req.time_stats,
    )
    req.tokenizer = self.tokenizer
    ...
    self.init_req_max_new_tokens(req)
    error_msg = validate_input_length(
        req,
        self.max_req_input_len,
        self.server_args.allow_auto_truncate,
    )
    ...
    added_to_grammar_queue = self.grammar_manager.process_req_with_grammar(req)
    if not added_to_grammar_queue:
        self._add_request_to_queue(req)
```

入队根据模式不同进入 waiting queue 或 disaggregation 队列。普通模式就是 `waiting_queue.append(req)`:

```python
# python/sglang/srt/managers/scheduler.py
def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
    if self.disaggregation_mode == DisaggregationMode.NULL:
        if not self._set_or_validate_priority(req):
            return
        if self._abort_on_queued_limit(req):
            return
        self._prefetch_kvcache(req)
        self.waiting_queue.append(req)
        req.time_stats.set_wait_queue_entry_time()
    elif self.disaggregation_mode == DisaggregationMode.PREFILL:
        ...
    elif self.disaggregation_mode == DisaggregationMode.DECODE:
        ...
```

## 6. Batch 生命周期: `Req -> ScheduleBatch -> ModelWorkerBatch -> ForwardBatch`

`Req` 保存单个请求的输入、输出、采样和 cache 状态:

```python
# python/sglang/srt/managers/schedule_batch.py
class Req(ReqDllmMixin):
    """The input and output status of a request."""

    def __init__(
        self,
        rid: str,
        origin_input_text: str,
        origin_input_ids: List[int],
        sampling_params: SamplingParams,
        return_logprob: bool = False,
        top_logprobs_num: int = 0,
        token_ids_logprob: List[int] = None,
        stream: bool = False,
        ...
    ):
        self.rid = rid
        self.origin_input_text = origin_input_text
        self.origin_input_ids = origin_input_ids
        self.output_ids = []
        self.fill_ids = []
        ...
        self.sampling_params = sampling_params
        self.custom_logit_processor = custom_logit_processor
```

`ScheduleBatch` 是 scheduler 管理的 batch:

```python
# python/sglang/srt/managers/schedule_batch.py
@dataclasses.dataclass
class ScheduleBatch(ScheduleBatchDisaggregationDecodeMixin):
    """Store all information of a batch on the scheduler."""

    reqs: List[Req]
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator = None
    tree_cache: BasePrefixCache = None
    forward_mode: ForwardMode = None
    sampling_info: SamplingBatchInfo = None
    input_ids: torch.Tensor = None
    req_pool_indices: torch.Tensor = None
    seq_lens: torch.Tensor = None
    out_cache_loc: torch.Tensor = None
```

batch 有两个基本 forward mode:

```python
# python/sglang/srt/model_executor/forward_batch_info.py
class ForwardMode(IntEnum):
    # Extend a sequence. The KV cache of the beginning part of the sequence is already computed.
    # It is also called "prefill" in common terminology.
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both EXTEND and DECODE when doing chunked prefill.
    MIXED = auto()
    # No sequence to forward.
    IDLE = auto()
```

`ScheduleBatch.prepare_for_extend()` 会把 prompt/prefill 所需的 token 拼成 tensor, 分配 KV cache 位置:

```python
# python/sglang/srt/managers/schedule_batch.py
def prepare_for_extend(self):
    self.forward_mode = ForwardMode.EXTEND

    reqs = self.reqs
    input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
    extend_num_tokens = sum(len(ids) for ids in input_ids)
    seq_lens = [len(r.fill_ids) for r in reqs]
    prefix_lens = [len(r.prefix_indices) for r in reqs]
    extend_lens = [r.extend_input_len for r in reqs]

    input_ids_tensor = torch.tensor(
        list(chain.from_iterable(input_ids)), dtype=torch.int64, pin_memory=_pin
    ).to(self.device, non_blocking=True)

    # Allocate memory
    out_cache_loc, req_pool_indices_tensor, req_pool_indices = alloc_for_extend(
        self
    )
```

`prepare_for_decode()` 则把上一轮输出 token 作为下一轮 decode 输入, 并为一个新 token 分配 KV cache 位置:

```python
# python/sglang/srt/managers/schedule_batch.py
def prepare_for_decode(self):
    self.forward_mode = ForwardMode.DECODE
    bs = len(self.reqs)

    # Update fields
    self.input_ids = self.output_ids
    self.output_ids = None

    # Allocate memory
    self.out_cache_loc = alloc_for_decode(self, token_per_req=1)

    # Update req-level memory management fields
    for req in self.reqs:
        req.decode_batch_idx += 1
        req.kv_committed_len += 1
        req.kv_allocated_len += 1
```

真正发给 model worker 的是 `ModelWorkerBatch`:

```python
# python/sglang/srt/managers/schedule_batch.py
def get_model_worker_batch(
    self, seq_lens_cpu_cache: Optional[torch.Tensor] = None
) -> ModelWorkerBatch:
    ...
    return ModelWorkerBatch(
        forward_mode=self.forward_mode,
        input_ids=self.input_ids,
        req_pool_indices=self.req_pool_indices,
        seq_lens=self.seq_lens,
        out_cache_loc=self.out_cache_loc,
        seq_lens_cpu=seq_lens_cpu,
        return_logprob=self.return_logprob,
        top_logprobs_nums=self.top_logprobs_nums,
        token_ids_logprobs=self.token_ids_logprobs,
        sampling_info=self.sampling_info,
        input_embeds=self.input_embeds,
        spec_algorithm=self.spec_algorithm,
        reqs=self.reqs,
        has_grammar=self.has_grammar,
    )
```

`ForwardBatch` 是 `ModelRunner` 使用的低层 GPU tensor 视图。文件头部直接写了三者关系:

```python
# python/sglang/srt/model_executor/forward_batch_info.py
"""
The following is the flow of data structures for a batch:

ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

- ScheduleBatch is managed by `scheduler.py::Scheduler`.
  It contains high-level scheduling data. Most of the data is on the CPU.
- ModelWorkerBatch is managed by `tp_worker.py::TpModelWorker`.
  It is a subset of `ScheduleBatch` that only contains data related to the model forward on GPU.
  It will be transformed from CPU scheduler to GPU model runner.
- ForwardBatch is managed by `model_runner.py::ModelRunner`.
  It contains low-level tensor data. Most of the data consists of GPU tensors.
"""
```

`ForwardBatch.init_new()` 会补齐 position、attention backend、KV pool、LoRA 等模型 forward 所需字段:

```python
# python/sglang/srt/model_executor/forward_batch_info.py
@classmethod
def init_new(cls, batch: ModelWorkerBatch, model_runner: ModelRunner):
    ret = cls(
        forward_mode=batch.forward_mode,
        batch_size=len(batch.seq_lens),
        input_ids=batch.input_ids,
        req_pool_indices=batch.req_pool_indices,
        seq_lens=batch.seq_lens,
        out_cache_loc=batch.out_cache_loc,
        mm_inputs=batch.multimodal_inputs,
        sampling_info=batch.sampling_info,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool=model_runner.token_to_kv_pool,
        attn_backend=model_runner.attn_backend,
        lora_ids=batch.lora_ids,
        input_embeds=batch.input_embeds,
    )
    ...
    if ret.forward_mode.is_decode() or ret.forward_mode.is_target_verify():
        if ret.positions is None:
            ret.positions = clamp_position(batch.seq_lens)
    else:
        positions, ret.extend_start_loc = compute_position(
            model_runner.server_args.attention_backend,
            ret.extend_prefix_lens,
            ret.extend_seq_lens,
            ret.extend_num_tokens,
        )
        if ret.positions is None:
            ret.positions = positions
```

## 7. Scheduler 如何选择 prefill 还是 decode

`get_next_batch_to_run()` 是调度核心。它会先把上一轮 prefill 完的请求合入 `running_batch`, 再尝试从 waiting queue 取新请求做 prefill; 如果没有新的 prefill batch, 就更新 running batch 做 decode:

```python
# python/sglang/srt/managers/scheduler.py
def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
    self._abort_on_waiting_timeout()
    self._abort_on_running_timeout()

    # Merge the prefill batch into the running batch
    if (
        not self.enable_hisparse
        and self.last_batch
        and self.last_batch.forward_mode.is_extend()
    ):
        self.last_batch.filter_batch(...)
        if not self.last_batch.is_empty():
            if self.running_batch.is_empty():
                self.running_batch = self.last_batch
            else:
                self.running_batch.merge_batch(self.last_batch)

    new_batch = self.get_new_batch_prefill()

    if new_batch is not None:
        # Run prefill first if possible
        ret = new_batch
    else:
        # Run decode
        if (
            not self.running_batch.is_empty()
            and not self.running_batch.is_prefill_only
        ):
            self.running_batch = self.update_running_batch(self.running_batch)
            ret = self.running_batch if not self.running_batch.is_empty() else None
        else:
            ret = None

    if ret:
        set_schedule_time_batch(ret)

    return ret
```

`get_new_batch_prefill()` 会按调度策略计算优先级, 用 `PrefillAdder` 按内存、最大 token 数、最大请求数等约束挑选 waiting queue 中可运行的请求:

```python
# python/sglang/srt/managers/scheduler.py
def _get_new_batch_prefill_raw(...):
    if (self.running_batch.batch_is_full or len(self.waiting_queue) == 0) and self.chunked_req is None:
        return None

    running_bs = len(self.running_batch.reqs)
    self.policy.calc_priority(self.waiting_queue, self.running_batch)

    adder = PrefillAdder(
        self.page_size,
        self.tree_cache,
        self.token_to_kv_pool_allocator,
        self.running_batch,
        self.new_token_ratio,
        self.max_prefill_tokens,
        chunked_prefill_size,
        ...
    )

    for req in self.waiting_queue:
        ...
        # adder decides whether the request can run in this prefill batch.
```

## 8. 模型 forward: Scheduler -> TpModelWorker -> ModelRunner -> Model

Scheduler 初始化模型 worker:

```python
# python/sglang/srt/managers/scheduler.py
def init_tp_model_worker(self):
    worker_kwargs = dict(
        server_args=self.server_args,
        gpu_id=self.gpu_id,
        tp_rank=self.tp_rank,
        moe_ep_rank=self.moe_ep_rank,
        pp_rank=self.pp_rank,
        attn_cp_rank=self.attn_cp_rank,
        moe_dp_rank=self.moe_dp_rank,
        dp_rank=self.dp_rank,
        nccl_port=self.nccl_port,
    )

    from sglang.srt.managers.tp_worker import TpModelWorker
    self.tp_worker = TpModelWorker(**worker_kwargs)

def init_model_worker(self):
    self.init_tp_model_worker()
    self.maybe_init_draft_worker()

    if self.spec_algorithm.is_none():
        self.model_worker = self.tp_worker
    else:
        self.model_worker = self.draft_worker
```

`TpModelWorker` 初始化 `ModelRunner`:

```python
# python/sglang/srt/managers/tp_worker.py
class TpModelWorker(BaseTpWorker):
    """A tensor parallel model worker."""

    def __init__(...):
        self._init_model_config()
        self._init_model_runner()
        ...

    def _init_model_runner(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        self._model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=self.server_args.mem_fraction_static,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            moe_ep_rank=self.moe_ep_rank,
            moe_ep_size=self.ep_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            nccl_port=self.nccl_port,
            dp_rank=self.dp_rank,
            server_args=self.server_args,
            ...
        )
```

Scheduler 的 `run_batch()` 将 batch 发给 model worker:

```python
# python/sglang/srt/managers/scheduler.py
def run_batch(self, batch: ScheduleBatch, pp_proxy_tensors=None):
    """Run a batch."""
    self.forward_ct += 1
    batch.forward_iter = self.forward_ct

    if self.is_generation:
        worker_batch_or_batch = batch.get_model_worker_batch()
        batch_result = self.model_worker.forward_batch_generation(
            worker_batch_or_batch,
            **kwargs,
        )
        future_indices_or_next_token_ids = batch_result.next_token_ids
        self.update_cache_from_scheduler(batch, batch_result)

        batch.output_ids = future_indices_or_next_token_ids
        ret = batch_result
    else:
        ...

    return ret
```

`TpModelWorker.forward_batch_generation()` 中的主路径是:

1. 从 `ModelWorkerBatch` 构造 `ForwardBatch`
2. 调 `ModelRunner.forward()`
3. 在最后一个 PP rank 上调用 `ModelRunner.sample()` 得到下一 token

```python
# python/sglang/srt/managers/tp_worker.py
def forward_batch_generation(
    self,
    model_worker_batch: ModelWorkerBatch,
    forward_batch: Optional[ForwardBatch] = None,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
    is_verify: bool = False,
    skip_attn_backend_init=False,
) -> GenerationBatchResult:
    if model_worker_batch is not None:
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)

    if self.pp_group.is_last_rank:
        out = self.model_runner.forward(
            forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
            skip_attn_backend_init=skip_attn_backend_init,
        )
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
            expert_distribution_metrics=out.expert_distribution_metrics,
            routed_experts_output=out.routed_experts_output,
            indexer_topk_output=out.indexer_topk_output,
        )

        if not model_worker_batch.is_prefill_only:
            # For normal requests, sample the next token ids.
            batch_result.next_token_ids = self.model_runner.sample(
                logits_output, forward_batch
            )
```

`ModelRunner.initialize()` 加载 sampler 和模型权重:

```python
# python/sglang/srt/model_executor/model_runner.py
def initialize(self, pre_model_load_memory: float):
    ...
    # Load the model
    self.sampler = create_sampler()
    self.load_model()
```

模型加载在 `load_model()`:

```python
# python/sglang/srt/model_executor/model_runner.py
def load_model(self):
    ...
    self.load_config = LoadConfig(
        load_format=self.server_args.load_format,
        download_dir=self.server_args.download_dir,
        model_loader_extra_config=self.server_args.model_loader_extra_config,
        tp_rank=self.tp_rank,
        ...
    )

    self.loader = get_model_loader(
        load_config=self.load_config,
        model_config=self.model_config,
    )
    self.model = self.loader.load_model(
        model_config=self.model_config,
        device_config=DeviceConfig(self.device, self.gpu_id),
    )
```

`ModelRunner.forward()` 先决定是否可用 CUDA graph, 然后按 `ForwardMode` 进入 decode 或 extend:

```python
# python/sglang/srt/model_executor/model_runner.py
def _forward_raw(self, forward_batch: ForwardBatch, ...):
    can_run_graph = bool(
        mode_check()
        and self.graph_runner
        and self.graph_runner.can_run(forward_batch)
    )

    if can_run_graph:
        ret = self.graph_runner.replay(
            forward_batch,
            skip_attn_backend_init=skip_attn_backend_init,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

    if forward_batch.forward_mode.is_decode():
        ret = self.forward_decode(...)
    elif forward_batch.forward_mode.is_split_prefill():
        ret = self.forward_split_prefill(...)
    elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
        ret, can_run_graph = self.forward_extend(...)
    elif forward_batch.forward_mode.is_idle():
        ret = self.forward_idle(...)
    else:
        raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

    return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)
```

decode forward 调模型的 `forward(input_ids, positions, forward_batch)`:

```python
# python/sglang/srt/model_executor/model_runner.py
def forward_decode(self, forward_batch: ForwardBatch, ...):
    if not skip_attn_backend_init:
        if hasattr(self.model, "prepare_forward_batch"):
            self.model.prepare_forward_batch(forward_batch)
        self.attn_backend.init_forward_metadata(forward_batch)

    return self.model.forward(
        forward_batch.input_ids,
        forward_batch.positions,
        forward_batch,
        **kwargs,
    )
```

extend/prefill forward 类似, 只是还要处理 input embeds、embedding override、piecewise cuda graph 等:

```python
# python/sglang/srt/model_executor/model_runner.py
def forward_extend(self, forward_batch: ForwardBatch, ...):
    kwargs = {}
    if forward_batch.input_embeds is not None:
        kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
    if not self.is_generation:
        kwargs["get_embedding"] = True

    if not skip_attn_backend_init:
        if hasattr(self.model, "prepare_forward_batch"):
            self.model.prepare_forward_batch(forward_batch)
        self.attn_backend.init_forward_metadata(forward_batch)

    ret = self.model.forward(
        forward_batch.input_ids,
        forward_batch.positions,
        forward_batch,
        **kwargs,
    )
    return (ret, can_run_graph)
```

## 9. 模型实现示例: Qwen2

以 `python/sglang/srt/models/qwen2.py` 为例, `Qwen2ForCausalLM` 包含 backbone、lm head 和 logits processor:

```python
# python/sglang/srt/models/qwen2.py
class Qwen2ForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.model = Qwen2Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(...)
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)
```

它的 `forward()` 先调用 `Qwen2Model`, 然后最后一个 PP rank 上走 lm head 和 logits processor:

```python
# python/sglang/srt/models/qwen2.py
@torch.no_grad()
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds: torch.Tensor = None,
    get_embedding: bool = False,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
) -> torch.Tensor:
    hidden_states = self.model(
        input_ids,
        positions,
        forward_batch,
        input_embeds,
        pp_proxy_tensors=pp_proxy_tensors,
    )

    if self.pp_group.is_last_rank:
        if not get_embedding:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.lm_head,
                forward_batch,
                aux_hidden_states,
            )
        else:
            return self.pooler(hidden_states, forward_batch)
    else:
        return hidden_states
```

backbone 的 forward 是标准 transformer block 循环, 每层拿 `forward_batch` 中的 attention/KV 元数据:

```python
# python/sglang/srt/models/qwen2.py
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds: torch.Tensor = None,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
) -> Union[torch.Tensor, PPProxyTensors]:
    if self.pp_group.is_first_rank:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds
        residual = None
    else:
        hidden_states = pp_proxy_tensors["hidden_states"]
        residual = pp_proxy_tensors["residual"]

    for i in range(self.start_layer, self.end_layer):
        layer = self.layers[i]
        hidden_states, residual = layer(
            positions,
            hidden_states,
            forward_batch,
            residual,
        )

    if self.pp_group.is_last_rank:
        hidden_states, _ = self.norm(hidden_states, residual)
    return hidden_states
```

单层中 self attention 和 MLP 的顺序:

```python
# python/sglang/srt/models/qwen2.py
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    residual: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Self Attention
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(
        positions=positions,
        hidden_states=hidden_states,
        forward_batch=forward_batch,
    )

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

## 10. 采样: SamplingParams, SamplingBatchInfo, Sampler

外部 sampling 参数会变成 `SamplingParams`:

```python
# python/sglang/srt/sampling/sampling_params.py
class SamplingParams:
    def __init__(
        self,
        max_new_tokens: int = 128,
        stop: Optional[Union[str, List[str]]] = None,
        stop_token_ids: Optional[List[int]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repetition_penalty: float = 1.0,
        min_new_tokens: int = 0,
        n: int = 1,
        json_schema: Optional[str] = None,
        regex: Optional[str] = None,
        ebnf: Optional[str] = None,
        ignore_eos: bool = False,
        ...
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature if temperature is not None else 1.0
        self.top_p = top_p if top_p is not None else 1.0
        self.top_k = top_k if top_k is not None else -1
        ...
        if 0 <= self.temperature < _SAMPLING_EPS:
            # top_k = 1 means greedy sampling
            self.temperature = 1.0
            self.top_k = 1
        if self.top_k == -1:
            self.top_k = TOP_K_ALL
```

Scheduler 建 batch 时会构造 `SamplingBatchInfo`, 把每个请求的 temperature/top-p/top-k/min-p 打成 GPU tensor, 同时初始化 penalty、grammar mask、logit bias 等:

```python
# python/sglang/srt/sampling/sampling_batch_info.py
@classmethod
def from_schedule_batch(cls, batch: ScheduleBatch, vocab_size: int):
    reqs = batch.reqs
    device = batch.device
    temperatures = torch.tensor(
        [r.sampling_params.temperature for r in reqs],
        dtype=torch.float,
        device=device,
    ).view(-1, 1)
    top_ps = torch.tensor(
        [r.sampling_params.top_p for r in reqs], dtype=torch.float, device=device
    )
    top_ks = torch.tensor(
        [r.sampling_params.top_k for r in reqs], dtype=torch.int32, device=device
    )
    min_ps = torch.tensor(
        [r.sampling_params.min_p for r in reqs], dtype=torch.float, device=device
    )
    ...
    ret = cls(
        temperatures=temperatures,
        top_ps=top_ps,
        top_ks=top_ks,
        min_ps=min_ps,
        is_all_greedy=all(r.sampling_params.top_k <= 1 for r in reqs),
        need_top_p_sampling=any(r.sampling_params.top_p != 1.0 for r in reqs),
        need_top_k_sampling=any(r.sampling_params.top_k != TOP_K_ALL for r in reqs),
        need_min_p_sampling=any(r.sampling_params.min_p > 0 for r in reqs),
        vocab_size=vocab_size,
        penalizer_orchestrator=penalizer_orchestrator,
        ...
    )
```

在 sample 前, `ModelRunner` 会先应用 grammar mask、penalty、logit bias:

```python
# python/sglang/srt/model_executor/model_runner.py
def _preprocess_logits(
    self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
):
    # Calculate logits bias and apply it to next_token_logits.
    sampling_info.update_regex_vocab_mask()
    sampling_info.apply_logits_bias(logits_output.next_token_logits)
    sampling_info.vocab_mask = None
```

`ModelRunner.sample()` 调用 `self.sampler`:

```python
# python/sglang/srt/model_executor/model_runner.py
def sample(
    self,
    logits_output: LogitsProcessorOutput,
    forward_batch: ForwardBatch,
) -> torch.Tensor:
    """Sample and compute logprobs and update logits_output."""
    self._preprocess_logits(logits_output, forward_batch.sampling_info)

    next_token_ids = self.sampler(
        logits_output,
        forward_batch.sampling_info,
        forward_batch.return_logprob,
        forward_batch.top_logprobs_nums,
        forward_batch.token_ids_logprobs,
        (
            forward_batch.positions
            if forward_batch.forward_mode.is_decode()
            else forward_batch.seq_lens - 1
        ),
    )
    self.maybe_update_ngram_token_table(next_token_ids, forward_batch)
    return next_token_ids
```

`Sampler.forward()` 的核心分支:

```python
# python/sglang/srt/layers/sampler.py
def forward(
    self,
    logits_output: LogitsProcessorOutput,
    sampling_info: SamplingBatchInfo,
    return_logprob: bool,
    top_logprobs_nums: List[int],
    token_ids_logprobs: List[List[int]],
    positions: torch.Tensor,
):
    logits = logits_output.next_token_logits
    logits = self._preprocess_logits(logits, sampling_info)

    if sampling_info.is_all_greedy:
        # Use torch.argmax if all requests use greedy sampling
        batch_next_token_ids = torch.argmax(logits, -1)
        if return_logprob:
            original_logprobs = logprobs = torch.nn.functional.log_softmax(
                logits, dim=-1
            )
    else:
        simple_sampling_case = (
            not sampling_info.need_top_p_sampling
            and not sampling_info.need_top_k_sampling
            and not sampling_info.need_min_p_sampling
        )
        ...
        logits.div_(sampling_info.temperatures)
        logits[:] = torch.softmax(logits, dim=-1)
        probs = logits

        batch_next_token_ids = self._sample_from_probs(
            probs, sampling_info, positions, simple_sampling_case
        )

    if return_logprob:
        self._attach_logprobs_to_output(...)

    self._sync_token_ids_across_tp(batch_next_token_ids, sampling_info)
    return batch_next_token_ids
```

## 11. 输出处理: token id 回到文本

Scheduler 拿到 `GenerationBatchResult` 后按 forward mode 分派:

```python
# python/sglang/srt/managers/scheduler.py
def process_batch_result(self, batch: ScheduleBatch, result):
    if batch.forward_mode.is_decode():
        self.process_batch_result_decode(batch, result)
    elif batch.forward_mode.is_extend():
        if batch.is_dllm():
            self.process_batch_result_dllm(batch, result)
        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.process_batch_result_disagg_prefill(batch, result)
        else:
            self.process_batch_result_prefill(batch, result)
    elif batch.forward_mode.is_prebuilt():
        self.process_batch_result_prebuilt(batch)
    elif batch.forward_mode.is_idle():
        self.process_batch_result_idle(batch, result)
```

prefill 结果会把本轮 sample 出的首个 output token 写回 `req.output_ids`, 检查是否完成, 未完成则缓存未完成请求:

```python
# python/sglang/srt/managers/scheduler_output_processor_mixin.py
def process_batch_result_prefill(self, batch, result):
    ...
    next_token_ids = next_token_ids.tolist()
    ...
    for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
        if req.finished() or req.is_retracted:
            continue

        if req.is_chunked <= 0:
            req.time_stats.set_prefill_finished_time()

            # req output_ids are set here
            req.output_ids.append(next_token_id)

            self._maybe_update_reasoning_tokens(req, next_token_id)

            req.check_finished()
            if req.finished():
                release_kv_cache(req, self.tree_cache)
                req.time_stats.set_completion_time()
            elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                maybe_cache_unfinished_req(req, self.tree_cache)
    ...
    self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)
```

decode 结果会追加每个请求的新 token, 更新 finish 状态和 logprob:

```python
# python/sglang/srt/managers/scheduler_output_processor_mixin.py
def process_batch_result_decode(self, batch, result):
    ...
    next_token_ids = next_token_ids.tolist()

    for i, req in enumerate(batch.reqs):
        ...
        next_token_id = next_token_ids[i]
        req.output_ids.append(next_token_id)

        self._maybe_update_reasoning_tokens(req, next_token_id)
        req.time_stats.set_last_decode_finish_time()
        req.check_finished(new_accepted_len)

        self._handle_finished_req(req, i, logits_output)
        ...
```

真正发送给 DetokenizerManager 的是 `BatchTokenIDOutput`:

```python
# python/sglang/srt/managers/scheduler_output_processor_mixin.py
def stream_output_generation(
    self,
    reqs: List[Req],
    return_logprob: bool,
    skip_req: Optional[Req] = None,
    is_idle_batch: bool = False,
):
    ...
    for req in reqs:
        ...
        decode_ids, read_offset = req.init_incremental_detokenize()
        decode_ids_list.append(decode_ids[req.send_decode_id_offset :])
        output_ids.append(output_ids_[send_token_offset:])
        ...

    # Send to detokenizer
    if reqs or is_idle_batch:
        self.send_to_detokenizer.send_output(
            BatchTokenIDOutput(
                rids=rids,
                finished_reasons=finished_reasons,
                decoded_texts=decoded_texts,
                decode_ids=decode_ids_list,
                read_offsets=read_offsets,
                output_ids=output_ids,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                ...
            )
        )
```

DetokenizerManager 从 Scheduler 收 `BatchTokenIDOutput`, decode 成字符串后发给 TokenizerManager:

```python
# python/sglang/srt/managers/detokenizer_manager.py
def event_loop(self):
    """The event loop that handles requests"""
    while True:
        recv_obj = self.recv_from_scheduler.recv_pyobj()
        output = self._request_dispatcher(recv_obj)
        if output is not None:
            self.send_to_tokenizer.send_pyobj(output)
```

decode 的核心是 `_decode_batch_token_id_output()`:

```python
# python/sglang/srt/managers/detokenizer_manager.py
def _decode_batch_token_id_output(self, recv_obj: BatchTokenIDOutput):
    ...
    # Decode token ids to strings
    if not self.disable_tokenizer_batch_decode:
        surr_texts = self._grouped_batch_decode(...)
        read_texts = self._grouped_batch_decode(...)
    else:
        surr_texts = [self.tokenizer.decode(...) for ...]
        read_texts = [self.tokenizer.decode(...) for ...]

    # Incremental decoding
    output_strs = []
    for i in range(bs):
        rid = recv_obj.rids[i]
        s = self.decode_status[rid]
        new_text = read_texts[i][len(surr_texts[i]) :]
        ...
        output_str = self.trim_matched_stop(
            s.decoded_text + new_text,
            recv_obj.finished_reasons[i],
            recv_obj.no_stop_trim[i],
        )

        # Incrementally send text.
        incremental_output = output_str[s.sent_offset :]
        s.sent_offset = len(output_str)
        output_strs.append(incremental_output)

    return output_strs
```

最后封装成 `BatchStrOutput`:

```python
# python/sglang/srt/managers/detokenizer_manager.py
def handle_batch_token_id_out(self, recv_obj: BatchTokenIDOutput):
    output_strs = self._decode_batch_token_id_output(recv_obj)
    return BatchStrOutput(
        rids=recv_obj.rids,
        http_worker_ipcs=recv_obj.http_worker_ipcs,
        finished_reasons=recv_obj.finished_reasons,
        output_strs=output_strs,
        output_ids=recv_obj.output_ids,
        prompt_tokens=recv_obj.prompt_tokens,
        completion_tokens=recv_obj.completion_tokens,
        cached_tokens=recv_obj.cached_tokens,
        ...
    )
```

TokenizerManager 收到后写入 `ReqState`, 唤醒 `_wait_one_response()`:

```python
# python/sglang/srt/managers/tokenizer_manager.py
async def _handle_batch_output(self, recv_obj):
    pending_notify: dict[str, ReqState] = {}
    for i, rid in enumerate(recv_obj.rids):
        state = self.rid_to_state.get(rid, None)
        ...
        meta_info = {
            "id": rid,
            "finish_reason": recv_obj.finished_reasons[i],
            "prompt_tokens": recv_obj.prompt_tokens[i],
            "weight_version": self.server_args.weight_version,
            "num_retractions": recv_obj.retraction_counts[i],
        }
        ...
        state.finished = recv_obj.finished_reasons[i] is not None
        if isinstance(recv_obj, BatchStrOutput):
            is_stream = getattr(state.obj, "stream", False)
            incremental = (
                self.server_args.incremental_streaming_output and is_stream
            )
            delta_text = recv_obj.output_strs[i]
            delta_output_ids = recv_obj.output_ids[i]
            output_offset = state.last_output_offset
            state.append_text(delta_text)
            state.output_ids.extend(delta_output_ids)

            if is_stream:
                if incremental:
                    output_token_ids = delta_output_ids
                    _slice_streaming_output_meta_info(meta_info, output_offset)
                    state.last_output_offset = len(state.output_ids)
                    out_dict = {
                        "text": delta_text,
                        "output_ids": output_token_ids,
                        "meta_info": meta_info,
                    }
                elif state.finished:
                    out_dict = {
                        "text": state.get_text(),
                        "output_ids": state.output_ids.copy(),
                        "meta_info": meta_info,
                    }
                else:
                    out_dict = {
                        "text": None,
                        "output_ids": state.output_ids,
                        "meta_info": meta_info,
                    }
            elif state.finished:
                out_dict = {
                    "text": state.get_text(),
                    "output_ids": state.output_ids.copy(),
                    "meta_info": meta_info,
                }
            else:
                out_dict = None

        if state.finished:
            del self.rid_to_state[rid]

        if out_dict is not None:
            state.out_list.append(out_dict)
            pending_notify[rid] = state

    for s in pending_notify.values():
        s.event.set()
```

## 12. 端到端时序图

```text
Client
  |
  | HTTP /generate or /v1/chat/completions
  v
http_server.py
  |
  | GenerateReqInput
  v
TokenizerManager.generate_request()
  |
  | normalize + tokenize + SamplingParams
  | TokenizedGenerateReqInput over ZMQ
  v
Scheduler.process_input_requests()
  |
  | TokenizedGenerateReqInput -> Req -> waiting_queue
  v
Scheduler.get_next_batch_to_run()
  |
  | Req -> ScheduleBatch
  | prepare_for_extend() or prepare_for_decode()
  v
Scheduler.run_batch()
  |
  | ScheduleBatch -> ModelWorkerBatch
  v
TpModelWorker.forward_batch_generation()
  |
  | ModelWorkerBatch -> ForwardBatch
  v
ModelRunner.forward()
  |
  | forward_extend() or forward_decode()
  v
model.forward(input_ids, positions, forward_batch)
  |
  | logits
  v
ModelRunner.sample()
  |
  | next_token_ids
  v
SchedulerOutputProcessorMixin
  |
  | update Req.output_ids, check finish
  | BatchTokenIDOutput over ZMQ
  v
DetokenizerManager
  |
  | token ids -> text
  | BatchStrOutput over ZMQ
  v
TokenizerManager.handle_loop()
  |
  | ReqState.out_list + event.set()
  v
HTTP/OpenAI/Python API response
```

## 13. 阅读主线建议

如果只想抓住推理主干, 建议按这个顺序读:

1. `python/sglang/srt/entrypoints/http_server.py`: `launch_server()`, `/generate`, `/v1/chat/completions`
2. `python/sglang/srt/managers/tokenizer_manager.py`: `generate_request()`, `_tokenize_one_request()`, `_send_one_request()`, `_wait_one_response()`, `handle_loop()`
3. `python/sglang/srt/managers/scheduler.py`: `event_loop_normal()`, `process_input_requests()`, `handle_generate_request()`, `get_next_batch_to_run()`, `run_batch()`, `process_batch_result()`
4. `python/sglang/srt/managers/schedule_batch.py`: `Req`, `ScheduleBatch.prepare_for_extend()`, `prepare_for_decode()`, `get_model_worker_batch()`
5. `python/sglang/srt/model_executor/forward_batch_info.py`: `ForwardMode`, `ForwardBatch.init_new()`
6. `python/sglang/srt/managers/tp_worker.py`: `TpModelWorker.forward_batch_generation()`
7. `python/sglang/srt/model_executor/model_runner.py`: `load_model()`, `forward()`, `forward_extend()`, `forward_decode()`, `sample()`
8. `python/sglang/srt/layers/sampler.py`: `Sampler.forward()`
9. `python/sglang/srt/managers/scheduler_output_processor_mixin.py`: `process_batch_result_prefill()`, `process_batch_result_decode()`, `stream_output_generation()`
10. `python/sglang/srt/managers/detokenizer_manager.py`: `event_loop()`, `_decode_batch_token_id_output()`, `handle_batch_token_id_out()`
