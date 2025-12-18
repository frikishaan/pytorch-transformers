import sentencepiece as spm

class Tokenizer:
    
    def __init__(self, path):
        self.sp = spm.SentencePieceProcessor()
        try:
            self.sp.load(path)
        except:
            raise Exception("Tokenizer not found at path: " + str(path))

    def encode(self, text):
        return self.sp.encode(text, out_type=int)
        
    def decode(self, ids):
        return self.sp.decode(ids).strip()

    def get_vocab_size(self):
        return self.sp.get_piece_size()

    def get_unk_id(self):
        return self.sp.unk_id()

    def get_pad_id(self):
        return self.sp.pad_id()

    def get_bos_id(self):
        return self.sp.bos_id()

    def get_eos_id(self):
        return self.sp.eos_id()

    def get_unk_token(self):
        return self.sp.unk_piece()

    def get_pad_token(self):
        return self.sp.pad_piece()

    def get_bos_token(self):
        return self.sp.bos_piece()

    def get_eos_token(self):
        return self.sp.eos_piece()