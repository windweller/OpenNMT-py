""" Onmt NMT Model base class definition """
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder, multigpu=False, lm_aux=False):
        # this argument is not really set even if we are training in distributed environment
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lm_aux = lm_aux
        self.src_generator = None

    def forward(self, src, tgt, lengths, dec_state=None, train=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
            lm_aux: whether to train encoder LM as auxiliary task
            train: no return of encoder states
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        # we still get memory bank (which is all hidden states)
        enc_final, memory_bank, lengths = self.encoder(src, lengths)

        enc_state = self.decoder.init_decoder_state(src, memory_bank, enc_final)
        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths=lengths)
        # decoder return "attns" for visualization purpose

        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None

        if self.lm_aux and train:
            # encoder states, src lengths, rest are the same!
            return memory_bank, lengths, decoder_outputs, attns, dec_state
        else:
            return decoder_outputs, attns, dec_state
