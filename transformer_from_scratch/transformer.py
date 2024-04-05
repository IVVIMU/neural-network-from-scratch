import torch
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(
            self,
            encoder,
            decoder,
            src_embed,
            trg_embed,
            src_pad_idx,
            trg_pad_idx,
            device
    ):
        """
        Args:
            encoder: encoder stack                    
            decoder: decoder stack
            src_embed: source embeddings and encodings
            trg_embed: target embeddings and encodings
            src_pad_idx: padding index          
            trg_pad_idx: padding index
            device: cuda or cpu
        
        Returns:
            output: sequences after decoder (batch_size, trg_seq_len, vocab_size)
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        """
        Args:
            src: raw sequences with padding (batch_size, seq_len)              
        
        Returns:
            src_mask: mask for each sequence (batch_size, 1, 1, seq_len)
        """
        # suppose src -> [batch_size, seq_len]
        # unsqueeze(1) src -> [batch_size, 1, seq_len]
        # unsqueeze(2) src -> [batch_size, 1, 1, seq_len]
        # attention score -> [batch_size, n_heads, seq_len, seq_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        """
        Args:
            trg: raw sequences with padding (batch_size, seq_len)              
        
        Returns:
            trg_mask: mask for each sequence (batch_size, 1, trg_seq_len, trg_seq_len)
        """
        # suppose trg -> [batch_size, seq_len]
        # trg_pad_mask -> [batch_size, 1, 1, seq_len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_seq_len = trg.shape[1]
        # trg_sub_mask -> [seq_len, seq_len]
        trg_sub_mask = torch.tril(torch.ones((trg_seq_len, trg_seq_len), device=self.device)).bool()
        # trg_pad_mask broadcast to [batch_size, 1, seq_len, seq_len]
        # duo to trg_pad_mask, trg_sub_mask broadcasting to [batch_size, 1, seq_len, seq_len]
        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask

    def forward(self, src, trg):
        """
        Args:
            trg: raw target sequences (batch_size, trg_seq_len)
            src: raw src sequences (batch_size, src_seq_len)
        
        Returns:
            output: sequences after decoder (batch_size, trg_seq_len, output_dim)
        """
        # create source and target masks 
        src_mask = self.make_src_mask(src)  # (batch_size, 1, 1, src_seq_len)
        trg_mask = self.make_trg_mask(trg)  # (batch_size, 1, trg_seq_len, trg_seq_len)

        src = self.encoder(self.src_embed(src), src_mask)  # (batch_size, src_seq_length, d_model)
        output = self.decoder(self.trg_embed(trg), src, trg_mask, src_mask)
        return output
