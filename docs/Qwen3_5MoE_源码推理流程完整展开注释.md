# Qwen3.5MoE 源码推理流程完整展开注释

## 0. 文档目标

这份文档针对以下源码目录做推理流程整理：

- `CLionProjects/transformers/src/transformers/models/qwen3_5_moe`

目标不是只停留在接口名，而是：

1. 按 **实际推理调用顺序** 展开源码。
2. 区分 `prefill` 和 `decode` 两个阶段。
3. 对关键源码尽量做 **逐行注释**。
4. 遇到内部调用时，继续向下展开。
5. 只要在 `transformers` 或可直接读取到的依赖中还能看到源码，就继续展开。

说明：

- 本文当前是 **第一阶段整理版**。
- 第一阶段优先覆盖 `transformers` 中 Qwen3.5MoE 的 **完整主调用流**。
- 后续还应继续补到 `torch.nn.Linear / Embedding / Conv1d / F.linear / softmax / topk / CacheLayer.update` 等更底层实现。

---

## 2. 相关源码文件

核心文件：

- `transformers/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py`
- `transformers/src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py`

向下展开时涉及：

- `transformers/src/transformers/masking_utils.py`
- `transformers/src/transformers/cache_utils.py`
- `transformers/src/transformers/generation/utils.py`

---

## 3. 总体推理流

### 3.1 纯文本模型

纯文本推理入口：

- `Qwen3_5MoeForCausalLM.forward`

主链：

```text
Qwen3_5MoeForCausalLM.forward
-> Qwen3_5MoeTextModel.forward
-> for each Qwen3_5MoeDecoderLayer.forward
   -> input_layernorm
   -> token mixer
      -> linear_attention: Qwen3_5MoeGatedDeltaNet.forward
      -> or full_attention: Qwen3_5MoeAttention.forward
   -> residual add
   -> post_attention_layernorm
   -> Qwen3_5MoeSparseMoeBlock.forward
      -> Qwen3_5MoeTopKRouter.forward
      -> Qwen3_5MoeExperts.forward
      -> shared_expert
   -> residual add
-> final norm
-> lm_head
```

### 3.2 多模态模型

多模态推理入口：

- `Qwen3_5MoeForConditionalGeneration.forward`

首轮 `prefill`：

```text
Qwen3_5MoeForConditionalGeneration.forward
-> Qwen3_5MoeModel.forward
   -> token embedding
   -> image/video features
      -> Qwen3_5MoeVisionModel.forward
   -> 用视觉特征替换 placeholder token 对应 embedding
   -> compute_3d_position_ids
   -> Qwen3_5MoeTextModel.forward
-> lm_head
```

增量 `decode`：

```text
generate()
-> prepare_inputs_for_generation()
   -> 非首轮时清掉 pixel_values / pixel_values_videos
   -> _prepare_position_ids_for_generation()
-> forward()
   -> Qwen3_5MoeModel.forward
   -> Qwen3_5MoeTextModel.forward
      -> past_key_values 生效
      -> linear_attention 进入 recurrent 分支
      -> full_attention 进入 cache update 分支
-> lm_head
```

---

## 4. 配置决定了推理形态

源码：`configuration_qwen3_5_moe.py`

```python
class Qwen3_5MoeTextConfig(PreTrainedConfig):
    model_type = "qwen3_5_moe_text"  # 文本子模型类型
    keys_to_ignore_at_inference = ["past_key_values"]  # 推理阶段输出里忽略缓存字段的兼容配置

    vocab_size: int = 248320  # 词表大小
    hidden_size: int = 2048  # 隐状态维度
    num_hidden_layers: int = 40  # decoder 层数
    num_attention_heads: int = 16  # full attention 的 query head 数
    num_key_value_heads: int = 2  # full attention 的 KV head 数
    hidden_act: str = "silu"  # 激活函数
    max_position_embeddings: int = 32768  # 最大位置长度
    rms_norm_eps: float = 1e-6  # RMSNorm epsilon
    use_cache: bool = True  # 允许缓存，decode 依赖它
    head_dim: int = 256  # full attention 单头维度
    linear_conv_kernel_dim: int = 4  # linear attention 中 depthwise conv 核大小
    linear_key_head_dim: int = 128  # linear attention key head dim
    linear_value_head_dim: int = 128  # linear attention value head dim
    linear_num_key_heads: int = 16  # linear attention key head 数
    linear_num_value_heads: int = 32  # linear attention value head 数
    moe_intermediate_size: int = 512  # 稀疏专家 FFN 中间维
    shared_expert_intermediate_size: int = 512  # shared expert 中间维
    num_experts_per_tok: int = 8  # 每个 token 选多少个专家
    num_experts: int = 256  # 总专家数
    output_router_logits: bool = False  # 是否输出路由 logits
    router_aux_loss_coef: float = 0.001  # 训练期路由辅助损失系数
    layer_types: list[str] | None = None  # 每层究竟是 linear_attention 还是 full_attention

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 0.25)  # 给 RoPE 一个默认部分旋转比例
        if self.layer_types is None:
            interval_pattern = kwargs.pop("full_attention_interval", 4)  # 默认每 4 层插一层 full attention
            self.layer_types = [
                "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)  # 继续走父类配置初始化
```

这里最重要的配置结论：

1. Qwen3.5MoE 不是每层都 full attention。
2. 大部分层是 `linear_attention`。
3. 每隔若干层插入一层 `full_attention`。
4. 每层 FFN 都是 `Sparse MoE + shared expert`。

---

## 5. Prefill 主入口

### 5.1 `Qwen3_5MoeForConditionalGeneration.forward`

这是多模态推理正向入口。

```python
def forward(
    self,
    input_ids: torch.LongTensor = None,  # 文本 token
    attention_mask: torch.Tensor | None = None,  # 文本 mask
    position_ids: torch.LongTensor | None = None,  # 位置 id，可外部传入
    past_key_values: Cache | None = None,  # KV / recurrent cache
    inputs_embeds: torch.FloatTensor | None = None,  # 可直接传 embedding
    labels: torch.LongTensor | None = None,  # 训练标签
    pixel_values: torch.Tensor | None = None,  # 图像 patch 输入
    pixel_values_videos: torch.FloatTensor | None = None,  # 视频 patch 输入
    image_grid_thw: torch.LongTensor | None = None,  # 图像的 T/H/W 网格
    video_grid_thw: torch.LongTensor | None = None,  # 视频的 T/H/W 网格
    mm_token_type_ids: torch.IntTensor | None = None,  # 多模态 token 类型
    logits_to_keep: int | torch.Tensor = 0,  # 仅保留末尾若干 token 的 logits
    **kwargs,
):
    outputs = self.model(  # 先进入多模态主模型
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )
    # 严格贴源展开见 5.2 节：`Qwen3_5MoeModel.forward`
    hidden_states = outputs[0]  # 取最后一层 hidden states

    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])  # 只对需要的位置做 vocab 投影

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

    aux_loss = None
    if kwargs.get("output_router_logits", False):
        aux_loss = load_balancing_loss_func(
            outputs.router_logits,
            self.config.text_config.num_experts,
            self.config.text_config.num_experts_per_tok,
            attention_mask,
        )
        if labels is not None:
            loss += self.config.text_config.router_aux_loss_coef * aux_loss.to(loss.device)

    return Qwen3_5MoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
        router_logits=outputs.router_logits,
    )
```

这里要注意：

1. 多模态 `prefill` 的核心不是“把图像和文本分别送入 LLM”，而是先把视觉编码变成 embedding。
2. 然后用这些 embedding 去替换文本序列里的图像/视频占位 token。
3. 最终送入 `language_model` 的仍然是一条统一的 embedding 序列。
4. 视觉分支只发生在首轮 `prefill`，正常 `decode` 阶段不会反复走这个视觉主干。

---

### 5.2 严格贴源展开：`Qwen3_5MoeModel.forward`

这是一段大模块源码，改为分节严格贴源展示；继续下钻请看：

- `5.3` `Qwen3_5MoeVisionModel.forward`
- `5.4` `compute_3d_position_ids`
- `5.5` `get_rope_index`

```python
def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    mm_token_type_ids: torch.IntTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple | Qwen3_5MoeModelOutputWithPast:
    r"""
    image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
        The temporal, height and width of feature shape of each image in LLM.
    video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
        The temporal, height and width of feature shape of each video in LLM.
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        image_outputs: BaseModelOutputWithPooling = self.get_image_features(
            pixel_values, image_grid_thw, return_dict=True
        )
        image_embeds = image_outputs.pooler_output
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        video_outputs: BaseModelOutputWithPooling = self.get_video_features(
            pixel_values_videos, video_grid_thw, return_dict=True
        )
        video_embeds = video_outputs.pooler_output
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if position_ids is None:
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            mm_token_type_ids=mm_token_type_ids,
        )

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        **kwargs,
    )

    return Qwen3_5MoeModelOutputWithPast(
        **outputs,
        rope_deltas=self.rope_deltas,
    )
```

### 5.3 严格贴源展开：`Qwen3_5MoeVisionModel.forward`

```python
def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            The temporal, height and width of feature shape of each image in LLM.

    Returns:
        `torch.Tensor`: hidden_states.
    """
    hidden_states = self.patch_embed(hidden_states)

    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
    hidden_states = hidden_states + pos_embeds

    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    for blk in self.blocks:
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    merged_hidden_states = self.merger(hidden_states)

    return BaseModelOutputWithPooling(
        last_hidden_state=hidden_states,
        pooler_output=merged_hidden_states,
    )
```

### 5.4 严格贴源展开：`compute_3d_position_ids`

继续下钻：`5.5` `get_rope_index`

```python
def compute_3d_position_ids(
    self,
    input_ids: torch.Tensor | None,
    inputs_embeds: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values: torch.Tensor | None = None,
    mm_token_type_ids: torch.IntTensor | None = None,
) -> torch.Tensor | None:
    past_key_values_length = 0 if past_key_values is None else past_key_values.get_seq_length()
    has_multimodal = image_grid_thw is not None or video_grid_thw is not None
    if has_multimodal and mm_token_type_ids is None and input_ids is not None:
        raise ValueError(
            "Multimodal data was passed (via `image_grid_thw` or `video_grid_thw`) but `mm_token_type_ids` is "
            "missing. Please pass `mm_token_type_ids` to the model so that multimodal RoPE (M-RoPE) can be "
            "computed correctly. `mm_token_type_ids` is returned by the processor alongside `input_ids`."
        )
    can_compute_mrope = input_ids is not None and mm_token_type_ids is not None and has_multimodal

    if can_compute_mrope and (self.rope_deltas is None or past_key_values_length == 0):
        position_ids, rope_deltas = self.get_rope_index(
            input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
        )
        self.rope_deltas = rope_deltas
    elif self.rope_deltas is not None and (past_key_values_length > 0 or input_ids is None):
        batch_size, seq_length, _ = inputs_embeds.shape
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            position_ids = position_ids.view(1, batch_size, -1).repeat(3, 1, 1).to(inputs_embeds.device)
        else:
            position_ids = torch.arange(past_key_values_length, past_key_values_length + seq_length)
            position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1).to(inputs_embeds.device)
        delta = self.rope_deltas.repeat_interleave(batch_size // self.rope_deltas.shape[0], dim=0)
        position_ids = position_ids + delta.to(device=inputs_embeds.device)
    else:
        position_ids = None
    return position_ids
```

### 5.5 严格贴源展开：`get_rope_index`

```python
def get_rope_index(
    self,
    input_ids: torch.LongTensor,
    mm_token_type_ids: torch.IntTensor,
    image_grid_thw: torch.LongTensor | None = None,
    video_grid_thw: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Difference from Qwen2VL/Qwen2.5VL's get_rope_index:
    - Since Qwen3.5 use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split too.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`):
            Token type ids matching each modality to a different value in the input sequence, i.e. text (0), image (1), video (2).
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """

    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1
    spatial_merge_size = self.config.vision_config.spatial_merge_size

    mrope_position_deltas = []
    position_ids = torch.zeros(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }

    for batch_idx, current_input_ids in enumerate(input_ids):
        input_token_type = mm_token_type_ids[batch_idx]
        if attention_mask is not None:
            current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
            input_token_type = input_token_type[attention_mask[batch_idx].bool()]

        input_type_group = []
        for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))

        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in input_type_group:
            if modality_type == 0:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                )
                current_pos += text_len
            else:
                grid_thw = next(grid_iters[modality_type])
                vision_position_ids = self.get_vision_position_ids(
                    current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
                )
                llm_pos_ids_list.append(vision_position_ids)
                current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        if attention_mask is not None:
            position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
        else:
            position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
        mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, mrope_position_deltas
```

---

## 6. 文本主干 `Qwen3_5MoeTextModel.forward`

这是最核心的 decoder 主循环。

```python
def forward(
    self,
    input_ids=None,  # token id 输入
    attention_mask=None,  # padding mask
    position_ids=None,  # 位置编码，可能是 4D 结构里的 text+vision
    past_key_values=None,  # cache
    inputs_embeds=None,  # 直接传 embedding
    use_cache=None,  # 是否启用 cache
    **kwargs,
):
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")  # 二选一

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)  # id -> embedding
        # 严格贴源展开见 6.1 节：`torch.nn.Embedding.forward`

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)  # 首轮 prefill 且要 cache 时，现场创建动态缓存
    # 严格贴源展开见 6.2 节：`DynamicCache.__init__`

    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)  # 固定扩成 4 路：text/t/h/w
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]  # full attention mask 使用文本位置
        position_ids = position_ids[1:]  # rotary embedding 使用 t/h/w 三路
    else:
        text_position_ids = None

    causal_mask = create_causal_mask(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=text_position_ids,
    )  # 给 full attention 层用
    # 严格贴源展开见 6.3 节：`create_causal_mask`
    linear_attn_mask = self._update_linear_attn_mask(attention_mask, past_key_values)  # 给 linear attention 层用

    hidden_states = inputs_embeds  # 残差流初始化
    position_embeddings = self.rotary_emb(hidden_states, position_ids)  # 生成 cos/sin

    for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        layer_mask = linear_attn_mask if self.config.layer_types[i] == "linear_attention" else causal_mask

        hidden_states = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=layer_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )  # 逐层推进

    hidden_states = self.norm(hidden_states)  # 最终 RMSNorm

    return Qwen3_5MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )
```

这里非常关键：

1. `full_attention` 层会需要标准 `key/value cache`。
2. `linear_attention` 层不是标准 KV cache，而是自己维护卷积状态和 recurrent state。
3. 所以 `DynamicCache` 必须按层类型建立不同的 cache layer。

---

### 6.1 严格贴源展开：`torch.nn.Embedding.forward`

```python
def forward(self, input: Tensor) -> Tensor:
    return F.embedding(
        input,
        self.weight,
        self.padding_idx,
        self.max_norm,
        self.norm_type,
        self.scale_grad_by_freq,
        self.sparse,
    )
```

### 6.2 严格贴源展开：`DynamicCache.__init__`

```python
def __init__(
    self,
    ddp_cache_data: Iterable[tuple[torch.Tensor | None, ...]] | None = None,
    config: PreTrainedConfig | None = None,
    offloading: bool = False,
    offload_only_non_sliding: bool = False,
):
    layers = []
    if config is not None:
        decoder_config = config.get_text_config(decoder=True)
        sliding_window = getattr(decoder_config, "sliding_window", None) or getattr(
            decoder_config, "attention_chunk_size", None
        )
        layer_types = getattr(decoder_config, "layer_types", None)
        if layer_types is None:
            layer_types = []
            for _ in range(decoder_config.num_hidden_layers):
                if sliding_window is not None:
                    layer_types.append("sliding_attention")
                else:
                    layer_types.append("full_attention")
        if hasattr(decoder_config, "num_kv_shared_layers"):
            layer_types = layer_types[: -decoder_config.num_kv_shared_layers]

        for layer_type in layer_types:
            if layer_type in ("sliding_attention", "chunked_attention"):
                layers.append(DynamicSlidingWindowLayer(sliding_window=sliding_window))
            elif layer_type in ("mamba", "conv", "linear_attention", "moe"):
                layers.append(LinearAttentionLayer())
            elif layer_type == "hybrid":
                layers.append(LinearAttentionAndFullAttentionLayer())
            else:
                layers.append(DynamicLayer())

    if ddp_cache_data is not None:
        for layer_idx, kv_and_optional_sliding in enumerate(ddp_cache_data):
            if config is None:
                sliding_window_tensor = kv_and_optional_sliding[2] if len(kv_and_optional_sliding) == 3 else None
                if sliding_window_tensor is not None:
                    sliding_window = sliding_window_tensor[0].item()
                    layers.append(DynamicSlidingWindowLayer(sliding_window=sliding_window))
                else:
                    layers.append(DynamicLayer())
            _, _ = layers[layer_idx].update(kv_and_optional_sliding[0], kv_and_optional_sliding[1])

    if len(layers) == 0:
        super().__init__(
            layer_class_to_replicate=DynamicLayer,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )
    else:
        super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)
```

### 6.3 严格贴源展开：`create_causal_mask`

```python
def create_causal_mask(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor | None = None,  # not used anymore but kept for BC
    *,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
) -> torch.Tensor | BlockMask | None:
    """
    Create a standard causal mask based on the attention implementation used (stored in the config). If `past_key_values`
    has an hybrid cache structure, this function will return the mask corresponding to one of the "full_attention" layers (to align
    to what is needed in the `modeling_xxx.py` files).
    """
    if not getattr(config, "is_causal", True):
        return create_bidirectional_mask(
            config,
            inputs_embeds,
            attention_mask,
            past_key_values=past_key_values,
            or_mask_function=or_mask_function,
            and_mask_function=and_mask_function,
        )

    if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(False)
    else:
        layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, q_length, kv_length, q_offset, kv_offset = (
        _preprocess_mask_arguments(config, inputs_embeds, attention_mask, past_key_values, position_ids, layer_idx)
    )
    if early_exit:
        return attention_mask

    batch_size, dtype, device = inputs_embeds.shape[0], inputs_embeds.dtype, inputs_embeds.device
    mask_factory_function = causal_mask_function
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]
    use_vmap = False

    if _is_torch_xpu_available:
        allow_is_causal_skip = not (getattr(past_key_values, "is_compileable", False) and q_length == 1)
    else:
        allow_is_causal_skip = not getattr(past_key_values, "is_compileable", False)

    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
        use_vmap = True
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False
        use_vmap = True

    if packed_sequence_mask is not None:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False

    causal_mask = mask_interface(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,
        dtype=dtype,
        config=config,
        use_vmap=use_vmap,
        device=device,
    )
    return causal_mask
```

---

## 7. 单层主逻辑 `Qwen3_5MoeDecoderLayer.forward`

```python
def forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    **kwargs,
):
    residual = hidden_states  # 第一条残差支路先保存输入

    hidden_states = self.input_layernorm(hidden_states)  # attention 前归一化

    if self.layer_type == "linear_attention":
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
        )  # 大多数层走 GatedDeltaNet
    elif self.layer_type == "full_attention":
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )  # 少数层走标准 full attention

    hidden_states = residual + hidden_states  # token mixer 之后第一次残差

    residual = hidden_states  # 第二条残差支路
    hidden_states = self.post_attention_layernorm(hidden_states)  # MoE 前归一化
    hidden_states = self.mlp(hidden_states)  # 这里的 mlp 实际是 Sparse MoE block
    if isinstance(hidden_states, tuple):
        hidden_states, _ = hidden_states
    hidden_states = residual + hidden_states  # MoE 之后第二次残差

    return hidden_states
```

这一层的结构非常清楚：

```text
residual
-> input_layernorm
-> token mixer (linear_attention / full_attention)
-> residual add
-> post_attention_layernorm
-> sparse moe
-> residual add
```

---

## 8. Full Attention 分支

### 8.1 `Qwen3_5MoeAttention.forward`

```python
def forward(
    self,
    hidden_states,
    position_embeddings,
    attention_mask,
    past_key_values=None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]  # (batch, seq)
    hidden_shape = (*input_shape, -1, self.head_dim)  # 方便后续 view 成多头

    query_states, gate = torch.chunk(
        self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
    )  # q_proj 一次性投出 query + gate 两部分
    gate = gate.reshape(*input_shape, -1)  # gate 会在输出端做 sigmoid 门控

    query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)  # q 做 head 内 RMSNorm
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # k 也做 norm
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # v 正常投影

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)  # q/k 施加 RoPE
    # 严格贴源展开见 8.2 节：`apply_rotary_pos_emb`

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)  # decode 时拼接缓存
        # 严格贴源展开见 8.3 节：`Cache.update`
        # 严格贴源展开见 8.4 节：`DynamicLayer.update`

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )  # 根据实现选择 eager/sdpa/flash

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )
    # 严格贴源展开见 8.5 节：`eager_attention_forward`

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()  # 合并 heads
    attn_output = attn_output * torch.sigmoid(gate)  # Qwen3.5 额外对注意力输出做门控

    attn_output = self.o_proj(attn_output)  # 输出投影回 hidden_size
    return attn_output, attn_weights
```
---

### 8.2 严格贴源展开：`apply_rotary_pos_emb`

```python
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Removes the interleaving of cos and sin from GLM
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed
```

### 8.3 严格贴源展开：`Cache.update`

继续下钻：`8.4` `DynamicLayer.update`

```python
def update(
    self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, *args, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
    """
    if self.layer_class_to_replicate is not None:
        while len(self.layers) <= layer_idx:
            self.layers.append(self.layer_class_to_replicate())

    if self.offloading:
        torch.cuda.default_stream(key_states.device).wait_stream(self.prefetch_stream)
        self.prefetch(layer_idx + 1, self.only_non_sliding)

    keys, values = self.layers[layer_idx].update(key_states, value_states, *args, **kwargs)

    if self.offloading:
        self.offload(layer_idx, self.only_non_sliding)

    return keys, values
```

### 8.4 严格贴源展开：`DynamicLayer.update`

```python
def update(
    self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Update the key and value caches in-place, and return the necessary keys and value states.
    """
    if not self.is_initialized:
        self.lazy_initialization(key_states, value_states)

    self.keys = torch.cat([self.keys, key_states], dim=-2)
    self.values = torch.cat([self.values, value_states], dim=-2)
    return self.keys, self.values
```

### 8.5 严格贴源展开：`eager_attention_forward`

```python
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights
```

---

## 9. Linear Attention 分支

### 9.1 `Qwen3_5MoeGatedDeltaNet.forward`

这个分支是 Qwen3.5MoE 非常关键的地方，也是大部分层真正使用的 token mixer。

```python
def forward(
    self,
    hidden_states,
    cache_params=None,
    attention_mask=None,
):
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)  # 先把 padding token 状态清零

    batch_size, seq_len, _ = hidden_states.shape  # 记录形状

    use_precomputed_states = (
        cache_params is not None and cache_params.has_previous_state(self.layer_idx) and seq_len == 1
    )  # decode 时单 token 且已有状态，进入 recurrent 快路径

    if use_precomputed_states:
        conv_state = cache_params.layers[self.layer_idx].conv_states  # 取卷积缓存
        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states  # 取递归状态缓存

    mixed_qkv = self.in_proj_qkv(hidden_states)  # 一次线性投影出 q/k/v
    # 严格贴源展开见 9.2 节：`torch.nn.Linear.forward`
    mixed_qkv = mixed_qkv.transpose(1, 2)  # 变成 conv1d 需要的 (batch, channels, seq)

    z = self.in_proj_z(hidden_states)  # RMSNormGated 里的 gate 分支 z
    z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

    b = self.in_proj_b(hidden_states)  # beta 门
    a = self.in_proj_a(hidden_states)  # a 用于构造 g 衰减项

    if use_precomputed_states:
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )  # decode 单步只更新最后一步卷积状态
    else:
        if cache_params is not None:
            conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)  # prefill 时把卷积缓存初始化好
            # 严格贴源展开见 9.3 节：`Cache.update_conv_state`
            # 严格贴源展开见 9.4 节：`LinearAttentionLayer.update_conv_state`
        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )  # 如果装了快路径库，就用 fused 实现
        else:
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])  # 否则用普通 Conv1d + SiLU
            # 严格贴源展开见 9.5 节：`torch.nn.Conv1d._conv_forward / forward`

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [self.key_dim, self.key_dim, self.value_dim],
        dim=-1,
    )  # 切回 q/k/v

    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()  # beta 门控
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)  # 构造连续时间衰减项
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)  # 头数对齐
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if not use_precomputed_states:
        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )  # prefill：分块并行版本
        # 严格贴源展开见 9.6 节：`torch_chunk_gated_delta_rule`
    else:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )  # decode：单步递归版本
        # 严格贴源展开见 9.7 节：`torch_recurrent_gated_delta_rule`

    if cache_params is not None:
        cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)  # 回写 recurrent state
        # 严格贴源展开见 9.8 节：`Cache.update_recurrent_state`
        # 严格贴源展开见 9.9 节：`LinearAttentionLayer.update_recurrent_state`

    core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
    z = z.reshape(-1, self.head_v_dim)
    core_attn_out = self.norm(core_attn_out, z)  # gated RMSNorm
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = self.out_proj(core_attn_out)  # 投影回 hidden_size
    return output
```

这段代码可以直接看出：

1. `prefill` 走 `chunk_gated_delta_rule`。
2. `decode` 走 `recurrent_gated_delta_rule`。
3. 也就是说，Qwen3.5 的 linear attention 先用分块并行算法完成整段 prompt，再在 decode 时退化成真正的递归更新。

---

### 9.2 严格贴源展开：`torch.nn.Linear.forward`

```python
def forward(self, input: Tensor) -> Tensor:
    """
    Runs the forward pass.
    """
    return F.linear(input, self.weight, self.bias)
```

### 9.3 严格贴源展开：`Cache.update_conv_state`

继续下钻：`9.4` `LinearAttentionLayer.update_conv_state`

```python
def update_conv_state(self, conv_states: torch.Tensor, layer_idx: int, **kwargs) -> torch.Tensor:
    """
    Updates the cache with the new `conv_states` for the layer `layer_idx`.
    """
    if not isinstance(self.layers[layer_idx], LinearAttentionCacheLayerMixin):
        raise ValueError("Cannot call `update_conv_state` on a non-LinearAttention layer!")
    conv_states = self.layers[layer_idx].update_conv_state(conv_states, **kwargs)
    return conv_states
```

### 9.4 严格贴源展开：`LinearAttentionLayer.update_conv_state`

```python
def update_conv_state(self, conv_states: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Update the linear attention cache in-place, and return the necessary conv states.
    """
    if not self.is_conv_states_initialized:
        self.lazy_initialization(conv_states=conv_states)

    if not self.has_previous_state:
        self.conv_states.copy_(conv_states)
        self.has_previous_state = True
    else:
        num_new_tokens = conv_states.shape[-1]
        if num_new_tokens >= self.conv_kernel_size:
            self.conv_states.copy_(conv_states[..., -self.conv_kernel_size :])
        else:
            new_conv_states = self.conv_states.roll(shifts=-num_new_tokens, dims=-1)
            new_conv_states[:, :, -num_new_tokens:] = conv_states
            self.conv_states.copy_(new_conv_states)

    return self.conv_states
```

### 9.5 严格贴源展开：`torch.nn.Conv1d._conv_forward / forward`

```python
def _conv_forward(self, input: Tensor, weight: Tensor, bias: Tensor | None):
    if self.padding_mode != "zeros":
        return F.conv1d(
            F.pad(
                input, self._reversed_padding_repeated_twice, mode=self.padding_mode
            ),
            weight,
            bias,
            self.stride,
            _single(0),
            self.dilation,
            self.groups,
        )

    return F.conv1d(
        input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    )

def forward(self, input: Tensor) -> Tensor:
    return self._conv_forward(input, self.weight, self.bias)
```

### 9.6 严格贴源展开：`torch_chunk_gated_delta_rule`

```python
def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
```

### 9.7 严格贴源展开：`torch_recurrent_gated_delta_rule`

```python
def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state
```

### 9.8 严格贴源展开：`Cache.update_recurrent_state`

继续下钻：`9.9` `LinearAttentionLayer.update_recurrent_state`

```python
def update_recurrent_state(self, recurrent_states: torch.Tensor, layer_idx: int, **kwargs) -> torch.Tensor:
    """
    Updates the cache with the new `recurrent_states` for the layer `layer_idx`.
    """
    if not isinstance(self.layers[layer_idx], LinearAttentionCacheLayerMixin):
        raise ValueError("Cannot call `update_conv_state` on a non-LinearAttention layer!")
    recurrent_states = self.layers[layer_idx].update_recurrent_state(recurrent_states, **kwargs)
    return recurrent_states
```

### 9.9 严格贴源展开：`LinearAttentionLayer.update_recurrent_state`

```python
def update_recurrent_state(self, recurrent_states: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Update the linear attention cache in-place, and return the necessary ssm states.
    """
    if not self.is_recurrent_states_initialized:
        self.lazy_initialization(recurrent_states=recurrent_states)
    self.recurrent_states.copy_(recurrent_states)
    return self.recurrent_states
```

---

## 10. MoE 分支

### 10.1 `Qwen3_5MoeSparseMoeBlock.forward`

```python
def forward(self, hidden_states: torch.Tensor):
    batch_size, sequence_length, hidden_dim = hidden_states.shape  # 先取原始形状
    hidden_states_reshaped = hidden_states.view(-1, hidden_dim)  # 把所有 token 拉平成二维
    shared_expert_output = self.shared_expert(hidden_states_reshaped)  # shared expert 总是执行
    # 严格贴源展开见 10.2 节：`Qwen3_5MoeMLP.forward`
    _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)  # router 选 top-k 专家
    # 严格贴源展开见 10.3 节：`Qwen3_5MoeTopKRouter.forward`
    expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)  # 稀疏专家执行
    # 严格贴源展开见 10.4 节：`Qwen3_5MoeExperts.forward`

    shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
    # shared expert 还要乘一个单独的 gate

    expert_output = expert_output + shared_expert_output  # 稀疏专家输出 + shared expert 输出
    expert_output = expert_output.reshape(batch_size, sequence_length, hidden_dim)  # 还原回 (batch, seq, hidden)
    return expert_output
```

Qwen3.5MoE 的 FFN 并不是只有 sparse experts，而是：

```text
sparse_expert_output + gated_shared_expert_output
```

---

### 10.2 严格贴源展开：`Qwen3_5MoeMLP.forward`

```python
def forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj
```

### 10.3 严格贴源展开：`Qwen3_5MoeTopKRouter.forward`

```python
def forward(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
    router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
    router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
    router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
    router_top_value = router_top_value.to(router_logits.dtype)
    router_scores = router_top_value
    return router_logits, router_scores, router_indices
```

### 10.4 严格贴源展开：`Qwen3_5MoeExperts.forward`

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up
        current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    return final_hidden_states
```

---

## 11. Decode 路径

`decode` 的关键不在 `forward` 主体改了很多，而在于：

1. `prepare_inputs_for_generation()` 对输入做裁剪。
2. 非首轮时，不再重复送视觉输入。
3. 位置编码改成增量位置。
4. `past_key_values` 驱动 `full_attention` 与 `linear_attention` 走缓存分支。

### 11.1 `prepare_inputs_for_generation`

```python
def prepare_inputs_for_generation(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    position_ids=None,
    use_cache=True,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    is_first_iteration=False,
    **kwargs,
):
    model_inputs = super().prepare_inputs_for_generation(
        input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        use_cache=use_cache,
        is_first_iteration=is_first_iteration,
        **kwargs,
    )  # 先走通用生成输入准备逻辑
    # 严格贴源展开见 11.1.1 节：`GenerationMixin.prepare_inputs_for_generation`

    if not is_first_iteration and use_cache:
        model_inputs["pixel_values"] = None  # decode 后续步不再重复送图像
        model_inputs["pixel_values_videos"] = None  # decode 后续步不再重复送视频

    return model_inputs
```

#### 11.1.1 严格贴源展开：`GenerationMixin.prepare_inputs_for_generation`

```python
def prepare_inputs_for_generation(
    self: "GenerativePreTrainedModel",
    input_ids: torch.LongTensor,
    next_sequence_length: int | None = None,
    past_key_values: Cache | None = None,
    attention_mask: torch.LongTensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    is_first_iteration: bool | None = False,
    **kwargs,
):
    model_inputs = {}

    input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
    if not self.config.is_encoder_decoder and inputs_embeds is not None and is_first_iteration:
        model_inputs[input_ids_key] = None
        prompt_embeds = (
            inputs_embeds[:, -next_sequence_length:, :] if next_sequence_length is not None else inputs_embeds
        )
        model_inputs["inputs_embeds"] = prompt_embeds.clone(memory_format=torch.contiguous_format)
        batch_size, sequence_length = prompt_embeds.shape[:2]
    else:
        input_ids = input_ids[:, -next_sequence_length:] if next_sequence_length is not None else input_ids
        model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
        batch_size, sequence_length = input_ids.shape[:2]

    if past_key_values is not None:
        model_inputs["past_key_values"] = past_key_values
    position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"
    if (position_ids := kwargs.pop(position_ids_key, None)) is not None:
        model_inputs[position_ids_key] = position_ids
    if (token_type_ids := kwargs.pop("token_type_ids", None)) is not None:
        model_inputs["token_type_ids"] = token_type_ids

    for model_input_name in [position_ids_key, "token_type_ids", "mm_token_type_ids"]:
        model_input = model_inputs.get(model_input_name)
        if model_input is not None and model_input.shape[-1] != sequence_length:
            model_input = model_input[..., -sequence_length:].clone(memory_format=torch.contiguous_format)
            model_inputs[model_input_name] = model_input

    encoder_attention_mask = attention_mask if self.config.is_encoder_decoder else None
    attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
    attention_mask = (
        kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
    )
    if (
        isinstance(past_key_values, Cache)
        and past_key_values.is_compileable
        and attention_mask is not None
        and attention_mask.ndim == 2
    ):
        causal_mask_creation_function = getattr(self, "create_masks_for_generate", create_masks_for_generate)
        attention_mask = causal_mask_creation_function(
            config=self.config,
            inputs_embeds=torch.empty((batch_size, sequence_length, 0), dtype=self.dtype, device=input_ids.device),
            attention_mask=attention_mask,
            past_key_values=model_inputs.get("past_key_values"),
            position_ids=model_inputs.get(position_ids_key),
            token_type_ids=model_inputs.get("token_type_ids"),
            mm_token_type_ids=model_inputs.get("mm_token_type_ids"),
            is_first_iteration=is_first_iteration,
        )

    if attention_mask is not None:
        model_inputs[attention_mask_key] = attention_mask

    if encoder_attention_mask is not None:
        model_inputs["attention_mask"] = encoder_attention_mask

    kwargs_to_avoid_forwarding = ("labels", "next_sequence_length")
    for key, value in kwargs.items():
        if key not in model_inputs and key not in kwargs_to_avoid_forwarding:
            model_inputs[key] = value

    if self.is_remote_code() and "cache_position" in set(inspect.signature(self.forward).parameters):
        logger.warning_once(
            "The remote code model you are currently using seems to expect `cache_position`. This arg has been "
            "removed from the Transformers library, and will stop being created in `generate` even for remote code models "
            "in a future release. Please open a PR on the remote code hub repo to remove any usage of `cache_position`."
        )
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(sequence_length, device=input_ids.device) + past_seen_tokens
        model_inputs["cache_position"] = cache_position

    return model_inputs
```

### 11.2 `_prepare_position_ids_for_generation`

```python
def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
    text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)  # 先准备文本增量位置

    past_length = 0
    if (cache := model_kwargs.get("past_key_values")) is not None:
        past_length = cache.get_seq_length()
    if past_length != 0 and self.model.rope_deltas is not None:
        position_ids = text_positions[None, ...] + self.model.rope_deltas  # decode 时直接加上 prefill 保存的 rope 偏移
        return position_ids

    if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
        inputs_tensor = model_kwargs["input_ids"]

    is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]
    if (
        is_input_ids
        and model_kwargs.get("mm_token_type_ids") is not None
        and (model_kwargs.get("image_grid_thw") is not None or model_kwargs.get("video_grid_thw") is not None)
    ):
        model_kwargs = {k: v for k, v in model_kwargs.items() if k != "input_ids"}
        vision_positions, rope_deltas = self.model.get_rope_index(inputs_tensor, **model_kwargs)  # 首轮或重新计算时，重建 3D 视觉位置
        self.model.rope_deltas = rope_deltas
    else:
        vision_positions = text_positions.unsqueeze(0).expand(3, -1, -1)  # 纯文本场景则 3 路视觉位置退化成文本位置
        self.model.rope_deltas = torch.zeros(
            inputs_tensor.shape[0], 1, dtype=torch.long, device=inputs_tensor.device
        )

    text_positions = text_positions[None, ...]
    position_ids = torch.cat([text_positions, vision_positions], dim=0)  # 拼成 (4, batch, seq)
    return position_ids
```

---

## 12. Prefill 与 Decode 的真正差异

### 12.1 相同点

- 都会进入 `Qwen3_5MoeModel.forward`
- 都会进入 `Qwen3_5MoeTextModel.forward`
- 都会逐层执行 `DecoderLayer`
- 都会经过 `MoE`

### 12.2 不同点

`prefill`：

- 一次输入整段 prompt
- 可能包含图像/视频
- `linear_attention` 走 `chunk_gated_delta_rule`
- `full_attention` 直接对整段序列做 attention
- 初始化并写入 cache

`decode`：

- 一次通常只输入 1 个新 token
- 非首轮时视觉输入会被清空，不再重复编码
- `linear_attention` 走 `recurrent_gated_delta_rule`
- `full_attention` 通过 `past_key_values.update()` 使用历史 KV
- 使用 `rope_deltas` 修正多模态位置

---

## 13. 本阶段结论

Qwen3.5MoE 的推理主线可以概括成：

```text
多模态 prefill:
视觉编码 -> 替换 placeholder embedding -> 计算 3D M-RoPE -> 文本 decoder 主干

文本 decoder 主干:
embedding -> [Norm -> TokenMixer(linear/full) -> residual -> Norm -> SparseMoE+SharedExpert -> residual] * N -> final norm

decode:
prepare_inputs_for_generation -> 只送新增 token -> 复用 cache + rope_deltas -> 重复 decoder 单步前向
```

Qwen3.5MoE 的两个最核心特征就是：

1. **Token mixer 混合了 linear attention 与 full attention**
2. **FFN 全面使用 Sparse MoE + shared expert**

---

## 14. 补充：`GenerationMixin.generate()` 外层循环

上面文档已经写了 Qwen3.5MoE 模型侧如何重写 `prepare_inputs_for_generation`；这里再把 `transformers/generation/utils.py` 中真正驱动 `prefill -> decode` 的外层循环补上。

```python
def generate(...):
    model_forward = (
        self.get_compiled_call(generation_config.compile_config)
        if self._valid_auto_compile_criteria(model_kwargs, generation_config)
        else self.__call__
    )  # 满足条件则拿编译后的 forward，否则直接用模型 __call__

    prefill_consumed = False
    outputs = self._prefill(
        input_ids,
        generation_config,
        model_kwargs,
        is_first_iteration=not generation_config.is_assistant,
    )  # 先执行首轮 prefill；这一轮通常会把整段 prompt 全部过一遍并建立 cache

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        if prefill_consumed:
            next_sequence_length = 1 if model_kwargs["use_cache"] else None  # decode 时通常只切最后 1 个 token
            model_inputs = self.prepare_inputs_for_generation(
                input_ids,
                next_sequence_length=next_sequence_length,
                **model_kwargs,
            )  # 这里会把 input_ids / cache / mask / position_ids 整理成下一轮 forward 的输入
            # 这里调用的仍然是 11.1.1 节同一个 `GenerationMixin.prepare_inputs_for_generation`

            with self._optimize_model_for_decode():
                outputs = model_forward(**model_inputs, return_dict=True)  # 真正执行单步 decode forward

        prefill_consumed = True
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )  # 把本轮产出的 cache / mask / 位置相关信息写回，供下一轮 while 继续用

        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
        next_token_scores = logits_processor(input_ids, next_token_logits)  # 对最后一个位置的 logits 做后处理

        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)  # greedy 时直接取最大分数 token

        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)  # 把新 token 接到当前序列尾部
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0  # 全部结束才会跳出 while

        del outputs
```

这也解释了为什么 Qwen3.5MoE 的 `decode` 看起来像“反复调用同一个 `forward`”：真正的循环并不写在模型文件里，而是写在 `GenerationMixin.generate()` 里。

---

## 15. 补充：`torch.nn.functional` 可展开边界

对于本文已经继续下钻到的 `Embedding.forward / Linear.forward / Conv1d.forward`，当前环境里都能读取到对应 Python 模块源码，所以已经补到了调用点下方。

但再往下一层的：

1. `F.embedding`
2. `F.linear`
3. `F.conv1d`
4. `softmax`
5. `topk`
6. `one_hot`
7. `masked_scatter`
8. `index_add_`

很多会继续落到 `ATen / torch._C / C++/CUDA` 内核实现；在当前这份本机 Python 环境里，不能像 `.py` 文件那样继续完整读取其底层源码体。所以这里明确把它们标记为“Python 调度层可见、底层算子内核不可直接展开”的边界。

如果还要继续往下补，下一轮最值得追加的是：

1. `Qwen3_5MoeVisionBlock / VisionAttention / PatchEmbed / PatchMerger` 的逐行展开
2. `eager_attention_forward / sdpa / flash-attn` 分发路径差异
3. 若本机有完整 PyTorch 源码树，再进一步补 `ATen` 对应算子入口
