import refinitiv.dataplatform.eikon as ek
import datetime
import pandas as pd



class eikon_api:
    """Class for accessing Eikon Data
       See https://developers.refinitiv.com/en/api-catalog/eikon/eikon-data-api
       
    Args:
        data_path (str): Data path to folder that contains the data files. If 
        left blank files will not be saved.
    """

    def __init__(self, data_path: str=None):
        ek.set_app_key('xxxxxxxxxxxxxx')
        self.data_path = data_path
    
    def getSecurity(ric: str) -> pd.DataFrame:
        """Gets security prices at minute intervals
        
        Returns:
        Prices (DataFrame): Date, HIGH, LOW, OPEN, CLOSE, COUNT, VOLUME
        """
        minute = ek.get_timeseries([ric],  
                        start_date='2021-11-01',  # Eikon API limit of last 2 years
                        end_date=datetime.timedelta(0), 
                        interval='minute')       
        if self.data_path != None:
            minute.to_csv(f'{self.data_path}minute_{ric}.csv')
        return minute

    def getNKY225Constituents() -> pd.DataFrame:
        """Gets list of Nikkei 225 constituents and corresponding industry sectors
        
        Returns:
            NKY_Constituents (DataFrame): List of Nikkei 225 constituent Instruments, Names,
            and TSE 33 Subsector Name
        """
        df_nikkei, err = ek.get_data(
            instruments = ['Index(NIK225)'],
            fields = ['TR.OrganizationName','TR.TSE33SectorNameMain','TR.TSE33SectorNameSub']
        )
        if self.data_path != None:
            df_nikkei.to_csv(f'{self.data_path}minute_{ric}.csv')
        return df_nikkei


if __name__ == '__main__':
    ea = eikon_api()  # folder to save data in
    ric = '6758.T' # Sony, 'MSFT.O' Microsoft, '.N225' Nikkei 225
    df = ea.getSecurity(ric)
    print(df.head())

    # # Load TSE 33 Industry Sectors into data/prices folder
    # ea = eikon_api("./prices/")  # folder to save data in
    # for ind in ["FISH", "FOOD", "MING", "PETE", "CNST", "METL", "GLSS", "TXTL", "PAPR", "CHEM", "PHAM", "RUBR", "TEQP", "STEL", "NFRO", "MCHN", 
    # "ELEC", "PRCS", "MISC", "COMS", "SVCS", "EPNG", "RAIL", "SHIP", "AIRL", "WHSE", "WHOL", "RETL", "BNKS", "SECU", "INSU", "FINS", "RLTY"]
    #     ea.getSecurity(f".I{ind}.T")

    # # Load Nikkei stock prices into data/prices folder
    # ea = eikon_api("./prices/")  # folder to save data in
    # for s in [1332, 3103, 5707, 6703, 6758, 7203, 6098, 9432]:
    #     ea.getSecurity(f"{s}.T")



