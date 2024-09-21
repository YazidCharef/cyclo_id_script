# Typhoon Classification Project

## Setup and Execution

1. Create and activate virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Update `config.py`:
   - Set `NETCDF_PATH` to your `Pacific_Pressure_750.nc` file path
   - Set `SST_FILE` to your `sst_data_1979_2021.nc` file path

4. Ensure `typhoon_data_Cyclogenesis_Identification.csv` is in the correct location

5. Run the script:
   ```
   python main.py
   ```

6. Check output files for results and plots

Note: If memory issues occur, reduce `BATCH_SIZE` in `config.py`.
