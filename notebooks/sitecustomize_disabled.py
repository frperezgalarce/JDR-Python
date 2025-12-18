
import os
os.environ.setdefault("RPY2_CFFI_MODE", "ABI")

try:
    import rpy2.rinterface_lib.openrlib as openrlib
    import rpy2.rinterface_lib.conversion as rconv

    def _cchar_to_str(c, encoding="cp1252", *args):
        return openrlib.ffi.string(c).decode(encoding)

    def _cchar_to_str_with_maxlen(c, maxlen, encoding="cp1252", *args):
        return openrlib.ffi.string(c, maxlen).decode(encoding)

    rconv._cchar_to_str = _cchar_to_str
    rconv._cchar_to_str_with_maxlen = _cchar_to_str_with_maxlen
except Exception:
    pass
