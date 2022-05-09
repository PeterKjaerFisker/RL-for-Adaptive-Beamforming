import h5py
import sys

cmd_input = sys.argv
if len(cmd_input) > 1:
    FILENAME = sys.argv[1]
    DECIMATOR = int(sys.argv[2])
else:
    FILENAME = "pedestrian_LOS_SARSA_TTFF_2-2-4-8-0-0_7000_10000_results.hdf5"
    DECIMATOR = 4

# %% main
if __name__ == "__main__":
    with h5py.File(f'Results/Centralized_Agent_Sweeps/HDF5/{FILENAME}', 'r+') as f_src:
        with h5py.File(f'Results/Centralized_Agent_Sweeps/HDF5/deci_{DECIMATOR}_{FILENAME}', 'a') as f_dst:
            for key, value in f_src.items():
                try:
                    del f_dst[f'{key}']
                    f_dst.create_dataset(f'/{key}', data=f_src[key][::DECIMATOR])
                except KeyError:
                    f_dst.create_dataset(f'/{key}', data=f_src[key][::DECIMATOR])
    print("done")
