import typing
from gtsam import symbol, symbolChr, symbolIndex


class KeyType:
    BUNDLE_T_MASTER_TAG_POSE = "t"
    BUNDLE_T_AID_TAG_POSE = "a"
    WORLD_T_CAMERA_POSE = "x"
    WORLD_T_TAG_POSE = "w"
    WORLD_T_POINTS = "p"
    WORLD_T_BUNDLE_POSE = "b"


class Key:

    def __init__(self, gtsam_chr, gtsam_id, params: dict) -> None:
        self.param = params
        self.gtsam_id = gtsam_id
        self.gtsam_chr = gtsam_chr
        for key in self.param:
            setattr(self, key, self.param[key])
            
    def __str__(self) -> str:
        _str = ""
        for key in self.param:
            _str += f"{key}_{self.param[key]}_"
        return _str

    def to_gtsam(self):
        return symbol(self.gtsam_chr, self.gtsam_id)


class KeyGenerator:
    class param:
        def __init__(self, name, bits_len, mask, shift) -> None:
            self.name = name
            self.len = bits_len
            self.mask = mask
            self.shift = shift
            

    def __init__(self, symbol_chr: str) -> None:
        self.param_list: list[KeyGenerator.param] = []
        self.n_param = 0
        self.symbo_chr = symbol_chr

    def _add_param(self, name, word_len):
        self.param_list.append(self.param(name, word_len, 0, 0))
        self.n_param += 1

    def _finish(self):
        self.param_list = sorted(
            self.param_list, key=lambda x: x.len, reverse=True)

        shift = 0
        for i in range(self.n_param):
            self.param_list[i].mask = (1 << self.param_list[i].len)-1
            self.param_list[i].shift = shift
            shift += self.param_list[i].len

    def id(self, params: dict):
        symbol_id = 0
        for i in range(self.n_param):
            param_name = self.param_list[i].name
            symbol_id |= params[param_name] << self.param_list[i].shift
        return symbol_id

    def parse_id(self, symbol_id):
        params = {}
        for i in range(self.n_param):
            params[self.param_list[i].name] = (
                symbol_id >> self.param_list[i].shift) & self.param_list[i].mask
        return params

    def _key(self, params: dict):
        return Key(self.symbo_chr, self.id(params), params)
    
    def from_gtsam(self, gtsam_key):
        char = f"{symbolChr(gtsam_key):c}"
        index = symbolIndex(gtsam_key)
        assert char == self.symbo_chr, "Not the same symbol type"
        return self._key(self.parse_id(gtsam_key))

class WorldMasterTagKeyGenerator(KeyGenerator):
    def __init__(self) -> None:
        super().__init__(KeyType.BUNDLE_T_MASTER_TAG_POSE)

        self._add_param("tag_id", 8)
        self._add_param("bundle_id", 8)
        self._add_param("camera_id", 8)
        self._finish()

    def get_key(self, tag_id, bundle_id, camera_id):
        return self._key({"tag_id": tag_id, "bundle_id": bundle_id, "camera_id": camera_id})

class WorldAidTagKeyGenerator(KeyGenerator):
    def __init__(self) -> None:
        super().__init__(KeyType.BUNDLE_T_AID_TAG_POSE)

        self._add_param("tag_id", 8)
        self._add_param("bundle_id", 8)
        self._add_param("camera_id", 8)
        self._finish()

    def get_key(self, tag_id, bundle_id, camera_id):
        return self._key({"tag_id": tag_id, "bundle_id": bundle_id, "camera_id": camera_id})

class WorldCameraPoseKeyGenerator(KeyGenerator):
    def __init__(self) -> None:
        super().__init__(KeyType.WORLD_T_CAMERA_POSE)

        self._add_param("camera_id", 8)
        self._finish()

    def get_key(self, camera_id):
        return self._key({"camera_id": camera_id})
    
class WorldBundlePoseKeyGenerator(KeyGenerator):
    def __init__(self) -> None:
        super().__init__(KeyType.WORLD_T_BUNDLE_POSE)

        self._add_param("bundle_id", 8)
        self._finish()

    def get_key(self, bundle_id):
        return self._key({"bundle_id": bundle_id})


class BundleMasterTagKeyGenerator(KeyGenerator):
    def __init(self):
        super().__init__(KeyType.BUNDLE_T_MASTER_TAG_POSE)

        self._add_param("tag_id", 8)
        self._finish()
    def get_key(self, tag_id):
        return self._key({"tag_id": tag_id})
    
class BundleAidTagKeyGenerator(KeyGenerator):
    def __init(self):
        super().__init__(KeyType.BUNDLE_T_AID_TAG_POSE)

        self._add_param("tag_id", 8)
        self._finish()
    def get_key(self, tag_id):
        return self._key({"tag_id": tag_id})


