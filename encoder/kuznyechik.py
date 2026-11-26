from . import Encoder
import os
import gostcrypto


class KuznyechikEncoder(Encoder):
    def __init__(self):
        self.cipher_obj = None

    def encrypt(self, text: bytes) -> bytes:
        if self.cipher_obj is None:
            self.cipher_obj = gostcrypto.gostcipher.new('kuznechik',
                                            self.key,
                                            gostcrypto.gostcipher.MODE_CBC,
                                            pad_mode=gostcrypto.gostcipher.PAD_MODE_1, init_vect=self.iv)

        return b"" + self.cipher_obj.encrypt(text)

    def generate_key(self, key: bytes):
        self.key = key[:32]
        self.cipher_obj = None

    def generate_iv(self, iv: bytes):
        self.iv = iv[:16]
        self.cipher_obj = None
