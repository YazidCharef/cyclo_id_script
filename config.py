import torch

NETCDF_PATH = '/home/yazid/Documents/stage_cambridge/project_1/Pacific_Pressure_750.nc'


TYPHOON_FILE = 'data/typhoon_data_Cyclogenesis_Identification.csv'
SST_FILE = 'data/sst_data_1979_2021.nc'

BATCH_SIZE = 64
TEST_SPLIT = 0.2
VAL_SPLIT = 0.2

NUM_EPOCHS = 50
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Years for training (None to use all available years)

START_YEAR = None
END_YEAR = None