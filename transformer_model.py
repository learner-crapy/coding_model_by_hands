import math
import torch
from torch import nn, Tensor


def _generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf')."""
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
    return mask


class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens.

    The positional encodings have the same dimension as the embeddings so that the two can be summed.
    This implementation is taken from the "Attention Is All You Need" paper.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Register as buffer so it will be part of the module's state but not a parameter
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """A tiny Transformer encoder-decoder model wrapper for demonstration purposes."""

    def __init__(
        self,
        input_dim: int,
        model_dim: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        output_dim: int | None = None,
        max_seq_len: int = 5000,
    ) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.embedding = nn.Embedding(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout, max_seq_len)
        self.transformer = nn.Transformer(
            d_model=model_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(model_dim, output_dim or input_dim)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
    ) -> Tensor:
        """Run the forward pass.

        Args:
            src: (S, N) tensor where S is source sequence length and N is batch size
            tgt: (T, N) tensor where T is target sequence length
        """
        src_emb = self.embedding(src) * math.sqrt(self.model_dim)
        tgt_emb = self.embedding(tgt) * math.sqrt(self.model_dim)
        src_emb = self.pos_encoder(src_emb)
        tgt_emb = self.pos_encoder(tgt_emb)

        output = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )
        return self.generator(output)


if __name__ == "__main__":
    # Example usage
    BATCH_SIZE = 2
    SRC_SEQ_LEN = 10
    TGT_SEQ_LEN = 9
    VOCAB_SIZE = 1000

    model = TransformerModel(input_dim=VOCAB_SIZE, model_dim=128, num_heads=4)

    # Generate random integer sequences as dummy data
    src = torch.randint(0, VOCAB_SIZE, (SRC_SEQ_LEN, BATCH_SIZE))  # (S, N)
    tgt = torch.randint(0, VOCAB_SIZE, (TGT_SEQ_LEN, BATCH_SIZE))  # (T, N)

    src_mask = None
    tgt_mask = _generate_square_subsequent_mask(TGT_SEQ_LEN)

    out = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
    print("Output shape:", out.shape)  # Expected (T, N, VOCAB_SIZE)