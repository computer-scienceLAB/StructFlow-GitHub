# Visualization Samples

Local-only, correctly paired RGB samples are prepared in this tree and excluded from Git by `.gitignore`.

| Dataset | Sample directory | Contents |
| --- | --- | --- |
| KITTI 2012 | `kitti2012/` | `000000_10/_11` left/right benchmark pair |
| KITTI 2015 | `kitti2015/benchmark/` | Requested benchmark frames, including `000018_09`, `000018_10`, `000020_10`, `000100_10`, `000104_12`, `000137_10`, `000147_07`, `000167_10` |
| KITTI Raw | `kitti2015/raw_drive/`, `raw_long_drive/` | Short sequence plus the 97-frame stereo sequence used by the README demo |
| Sintel | `sintel/alley_1/`, `temple_2/`, `bamboo_1/` | Requested frames with adjacent stereo frames |
| FlyingThings3D | `flyingthings3d/C_0001/` | Four consecutive stereo frames |
| Spring | `spring/0001/` | Four consecutive stereo frames |

Run the tested relative-path commands in [README-reproduce.md](../README-reproduce.md). Do not commit or redistribute these dataset-derived files unless the applicable dataset license permits it.
