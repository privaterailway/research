__all__ = [
    # original ciphers
    'aes', 'blowfish', 'triple_des', 'kuznyechik',
    # added ciphers
    'camellia', 'sm4', 'seed', 'aes_ctr', 'cast5', 'des', 'rc2', 'magma',
    'chacha20', 'xchacha20', 'salsa20', 'rc4',
]


class Encoder(object):
    def encrypt(self, text: bytes) -> bytes:
        raise NotImplementedError
    
    def generate_key(self, key: bytes):
        raise NotImplementedError

    def generate_iv(self, iv: bytes):
        raise NotImplementedError

    def generate(self, key: bytes, iv: bytes):
        self.generate_key(key)
        self.generate_iv(iv)
    

import encoder.aes as aes
import encoder.blowfish as twofish
import encoder.triple_des as triple_des
import encoder.kuznyechik as kuznyechik
import encoder.camellia as camellia
import encoder.sm4 as sm4
import encoder.seed as seed
import encoder.aes_ctr as aes_ctr
import encoder.cast5 as cast5
import encoder.des as des
import encoder.rc2 as rc2
import encoder.magma as magma
import encoder.chacha20 as chacha20
import encoder.xchacha20 as xchacha20
import encoder.salsa20 as salsa20
import encoder.rc4 as rc4

ENCODER_FACTORY = {
    # --- original ciphers ---
    'AES': lambda: aes.AesEncoder(),
    'Blowfish': lambda: twofish.BlowfishEncoder(),
    '3DES': lambda: triple_des.TripleDesEncoder(),
    'Kuznyechik': lambda: kuznyechik.KuznyechikEncoder(),
    # --- added block ciphers (CBC) ---
    'Camellia': lambda: camellia.CamelliaEncoder(),
    'SM4': lambda: sm4.Sm4Encoder(),
    'SEED': lambda: seed.SeedEncoder(),
    'CAST5': lambda: cast5.Cast5Encoder(),
    'DES': lambda: des.DesEncoder(),
    'RC2': lambda: rc2.Rc2Encoder(),
    'Magma': lambda: magma.MagmaEncoder(),
    # --- added CTR / stream ciphers ---
    'AES-CTR': lambda: aes_ctr.AesCtrEncoder(),
    'ChaCha20': lambda: chacha20.ChaCha20Encoder(),
    'XChaCha20': lambda: xchacha20.XChaCha20Encoder(),
    'Salsa20': lambda: salsa20.Salsa20Encoder(),
    'RC4': lambda: rc4.Rc4Encoder(),
}
