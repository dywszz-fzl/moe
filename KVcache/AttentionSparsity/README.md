
# H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models

```python
query_states = self.q_proj(hidden_states)  # (bsz, q_len, hidden_size)
key_states   = self.k_proj(hidden_states)  # (bsz, q_len, hidden_size)
value_states = self.v_proj(hidden_states)  # (bsz, q_len, hidden_size)

# reshape & transpose
query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
key_states   = key_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
value_states = value_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)

```
## 二、Q / K / V 的形状
| 名称             | 形状                                       | 含义                |
| -------------- | ---------------------------------------- | ----------------- |
| `query_states` | **(bsz, num_heads, q_tokens, head_dim)** | 当前 step 的所有 query |
| `key_states`   | **(bsz, num_heads, k_tokens, head_dim)** | 当前或历史所有 key       |
| `value_states` | 同上                                       | 当前或历史所有 value     |

## 三、计算注意力分数
```python
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
```
| 操作                                   | 输入形状                                                                | 输出形状                               |
| ------------------------------------ | ------------------------------------------------------------------- | ---------------------------------- |
| `key_states.transpose(2, 3)`         | (bsz, heads, k_tokens, head_dim) → (bsz, heads, head_dim, k_tokens) | —                                  |
| `matmul(query_states, key_states^T)` | (bsz, heads, q_tokens, head_dim) × (bsz, heads, head_dim, k_tokens) | ✅ (bsz, heads, q_tokens, k_tokens) |


> 在历史 KV cache 中，只保留最重要（heavy）和最近（recent）的 token，对其余 token 屏蔽（mask）。

| 属性                     | 含义                               |
| ---------------------- | -------------------------------- |
| `heavy_budget_ratio`   | 分配给“重要历史 token”的比例               |
| `recent_budget_ratio`  | 分配给“最近 token”的比例                 |
| `previous_scores`      | 累积的注意力得分（跨 token 步）              |
| `attention_masks_next` | 下一个 step 应用的注意力 mask             |
| `cache_budget`         | 当前允许保留的 token 总数（heavy + recent） |

## 1. 计算基础注意力
```python
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / sqrt(head_dim)
attn_weights = F.softmax(attn_weights, dim=-1)
```

## 2. 计算累积得分
```python
# attn_weights: (BS, heads, q_tokens, k_tokens)
current_scores_sum = attn_weights.sum(0).sum(1)  # → (heads, k_tokens)
```
- sum(0)：在 batch 维度求和 → (heads, q_tokens, k_tokens)

- sum(1)：在 q_token 维度求和 → (heads, k_tokens)

表示每个 head 对每个 key-token 的总关注度（跨所有 query）。

## 3. 累积历史得分
```python
if self.previous_scores is not None:
    current_scores_sum[:, :-1] += self.previous_scores
else:
    # 初始化预算
    self.heavy_budget = int(self.heavy_budget_ratio * current_scores_sum.shape[-1])
    self.recent_budget = int(self.recent_budget_ratio * current_scores_sum.shape[-1])
    self.cache_budget = self.heavy_budget + self.recent_budget
```
- 首次调用：计算 heavy / recent 的预算上限。

- 后续调用：把上一步的注意力得分累积（即对历史 key 的重要性持续加权）。

## 4. 构造缓存mask
```python
attn_mask = torch.ones(heads, k_tokens+1)
if attn_tokens_all > self.cache_budget:
    # 启用 recent 策略
    attn_mask[:, :-self.recent_budget] = 0
    selected_set = self.previous_scores[:, :-self.recent_budget]

    # 启用 heavy 策略（选取 topk）
    _, keep_topk = selected_set.topk(k=self.heavy_budget, dim=-1)
    attn_mask = attn_mask.scatter(-1, keep_topk, 1)

```
- 先只保留 最近 recent_budget 个 token；
- 然后再在更早的部分里选出 历史 top-k（heavy） token；
- 最终形成一个形如 (heads, k_tokens+1) 的 mask。

## 5. 应用 mask 到下一步
```python
self.attention_masks_next = attn_mask.clone().unsqueeze(0).unsqueeze(2)
```
## 6. 更新累积得分
```python
score_mask = attn_mask[:, :-1]
score_mask[:, -self.recent_budget:] = 1
self.previous_scores = self.previous_scores * score_mask
```
