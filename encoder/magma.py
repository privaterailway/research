from . import Encoder
import gostcrypto


class MagmaEncoder(Encoder):
    """GOST R 34.12-2015 Magma, 256-bit key, CBC mode, 64-bit block."""

    def __init__(self):
        self.cipher_obj = None

    def encrypt(self, text: bytes) -> bytes:
        assert len(text) % 8 == 0, "Text length must be a multiple of 8 bytes for Magma"

        if self.cipher_obj is None:
            self.cipher_obj = gostcrypto.gostcipher.new('magma',
                                                        self.key,
                                                        gostcrypto.gostcipher.MODE_CBC,
                                                        pad_mode=gostcrypto.gostcipher.PAD_MODE_1,
                                                        init_vect=self.iv)

        return b"" + self.cipher_obj.encrypt(text)

    def generate_key(self, key: bytes):
        self.key = key[:32]
        self.cipher_obj = None

    def generate_iv(self, iv: bytes):
        self.iv = iv[:16]
        self.cipher_obj = None
