## 1. 经典馈神经网络FFN with GLU

![](https://skojiangdoc.oss-cn-beijing.aliyuncs.com/2024LLM/18.png)

左边的图像展示了经典 Transformer 中的前馈神经网络结构。输入经过一个线性层将维度从 `d_model` 映射到更高的隐藏维度 `hidden_dim`，然后通过 ReLU 激活，再经过另一个线性层将维度映射回 `d_model`，最后通过 Dropout 输出。这种设计为输入数据提供了非线性变换和基本的正则化。

右边的图像展示了 GPT 模型中的前馈神经网络结构，采用了门控线性单元（GLU）。输入首先通过一个线性层映射到 `2 * hidden_dim`，然后分成两部分，其中一部分经过 sigmoid 激活作为门控信号，与另一部分相乘形成门控后的输出。接着通过另一个线性层映射回 `d_model` 维度并输出。GLU 的设计使网络能够选择性地传递信息，提升模型的灵活性和表达能力。

**门控线性单元（Gated Linear Unit，GLU）** 是一种通过门控机制控制信息流的神经网络单元。在 GLU 中，输入被分成两个部分：一部分作为主输出，另一部分作为门控。通过对门控部分进行激活，然后与主输出相乘，GLU 能够控制信息流动，从而提高网络的表示能力和选择性。

假设输入向量 $ x $ 的维度为 $ d $，则 GLU 会将输入维度扩展为 $ 2d $ 以便分为两部分。GLU 的公式如下：
$$
\text{GLU}(x) = \left( x_1 \right) \times \sigma(x_2)
$$

其中：

- $ x_1 $ 和 $ x_2 $ 是输入向量 $ x $ 的两个分量，分别对应主输出和门控部分，维度均为 $ d $。
- $ \sigma(x_2) $ 是对 $ x_2 $ 应用 **Sigmoid 激活函数**，产生一个范围在 $[0, 1]$ 的门控值，用来控制信息的流动。
- $ x_1 \times \sigma(x_2) $ 表示按元素相乘，产生 GLU 的输出。

在代码实现中，通常会通过一个线性层将输入映射到 `2 * d` 维度，之后再将输出拆分为两部分分别作为 $ x_1 $ 和 $ x_2 $。

- **GLU 的作用**

1. **信息控制**：GLU 通过 Sigmoid 门控，使网络能够选择性地传递信息。Sigmoid 激活会使某些信息通过门控输出，而另一些信息被抑制或完全阻止。
2. **提高模型的非线性表达能力**：相比于普通的线性单元，GLU 能引入更丰富的非线性关系，提升模型的表现力。
3. **降低计算复杂度**：GLU 的结构简单，仅在主输出上应用门控操作，计算开销低，适合在深层网络中使用。

GLU 在自然语言处理、推荐系统和其他深度学习任务中较为常用，能够在不显著增加计算负担的情况下提升模型性能。

## 2. LLAMA中的门控前馈神经网络FFN with SwiGLU
![](https://skojiangdoc.oss-cn-beijing.aliyuncs.com/2024LLM/15.png)

> **llama中的前馈神经网络**
$$
\text{Output} = \text{Linear2}
\left( \textcolor{red}{\text{Activation}}\left( \textcolor{green}{\text{Linear1}}(x) \right) 
\odot \textcolor{gold}{\text{Linear3}}(x) \right)
$$
  它通过两个线性层（`Linear1`和`Linear3`），从`Linear1`输出的结果经过silu激活函数后，与`Linear3`输出的结果进行逐元素乘法，然后通过另一个线性层（`Linear2`）。

为什么llama要做这样的修改呢？为了要了解这个前馈网络的机制，我们先要了解一下SwiGLU激活函数。

- **SwiGLU（Switch-Gated Linear Unit）门控线性单元激活函数**

**SwiGLU** 是一种新型的激活函数，由 **Shazeer (2020)** 在论文 *“Gated Linear Units for Efficient and Scalable Deep Learning”* 中提出。它被用在 **深度学习模型的前馈神经网络（FFN）层**中，如 **LLaMA**、**GPT-3** 和其他大型 Transformer 模型中。SwiGLU 的设计核心是基于**门控机制（gating mechanism）**，它通过引入两个线性路径的输出，并结合逐元素乘法，实现了对信息的动态控制。这种门控结构类似于在 LSTM 和 GRU 等门控循环网络中的思想，但它被应用在 Transformer 的前馈网络（FFN）层中，用于增强网络的非线性表达能力和训练效率。

SwiGLU 激活函数的基本形式如下：

$$
\text{SwiGLU}(x) = \textcolor{red}{\text{GELU}}\left( \textcolor{green}{W_1^a} \cdot x \right) 
\odot \textcolor{gold}{W_1^b} \cdot x
$$

其中：
- $W_1^a$ 和 $W_1^b$ 是线性变换（全连接层）。
- $\odot$ 表示 **逐元素乘法**（element-wise multiplication）。
- **GELU**（Gaussian Error Linear Unit）是一个非线性激活函数，它与ReLU激活函数类似，但它比 ReLU 更平滑，适用于深度模型。

## 3. 混合专家模型
<center><img src="https://skojiangdoc.oss-cn-beijing.aliyuncs.com/2024LLM/16.png" alt="描述文字" width="400">

- **MoE 的核心公式：**
$$
\text{Output} = \sum_{i=1}^N G_i(x) \cdot E_i(x)
$$
> - **$E_i(x)$**：第 $i$ 个专家模型的输出。
> - **$G_i(x)$**：由路由器（Gate）计算得到的权重，决定哪些专家应该被激活、每个专家被激活的程度有多大。
> - **N**：专家模型的总数。通常来说，我们不会采用全部的专家的结果，而是采用权重最大的top-k个专家的结果，因此在实际计算时，N往往会被k所替代。

在混合专家模型的世界中、我们可以针对一个序列、一张表单或任意单独的一段信息设置一扇“门”（一个权重），我们也可以针对每一个token设置一个权重，**大部分用于大模型的MOE使用的是token-level的路由设置**。当我们将文字序列输入MOE时，常规的数据流如下所示 ↓ 

![](https://skojiangdoc.oss-cn-beijing.aliyuncs.com/2024LLM/training/107_.png)

![](https://skojiangdoc.oss-cn-beijing.aliyuncs.com/2024LLM/training/110.png)
### DeepSeekMoE的专家并行
专家并行机制是 **Mixture-of-Experts (MoE)** 模型在大规模分布式计算中的关键技术，旨在通过将模型的多个专家分布在不同的计算节点或设备上，来实现高效的计算和负载平衡。在大规模 **MoE** 模型中，模型的参数量非常庞大，且每个输入数据仅激活部分专家。如果每个设备都计算所有专家，计算量和内存消耗将急剧增加。专家并行机制的目标是将专家分配到多个设备上，只有特定的设备负责处理某些专家，从而达到：

- **负载均衡**：不同的设备处理不同的专家，避免某个设备负载过重。
- **计算效率**：通过并行计算，提高计算效率。
- **内存管理**：每个设备只需要加载和处理自己负责的专家，降低内存消耗。




