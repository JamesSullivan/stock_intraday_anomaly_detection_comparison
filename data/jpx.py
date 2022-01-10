import numpy as np
from numpy.lib import index_tricks
from numpy.lib.function_base import diff, interp
import pandas as pd
from datetime import datetime
from datetime import timedelta
from pprint import pprint
import matplotlib.pyplot as plt
from scipy import stats

try:
    import google.colab
    IN_COLAB = True
    DATA_PATH = "https://raw.githubusercontent.com/JamesSullivan/ad_test/main"
except BaseException:
    IN_COLAB = False
    DATA_PATH = "."

def minmax(data):
    min, max = data.min(), data.max()
    return (data - min) / (max - min)

def daily_minutes(start_date="11/01/21"):
    morning = pd.date_range(start_date, periods=150, freq="T")
    return morning.append(pd.date_range("11/01/21 03:30:00", periods=150, freq="T"))

def test_0to1(the_seed=0):
    np.random.seed(the_seed)
    return [np.random.rand() for i in range(0,300)]
    
class TOPIX_SECTORS:
    """A helper class to load TOPIX Sector Prices by name or code
    https://www.jpx.co.jp/english/markets/indices/line-up/files/e_fac_13_sector.pdf
    """

    _NAMES = ["Fishery, Agriculture & Forestry", "Foods", "Mining", "Oil & Coal Products", "Construction", "Metal Products", "Glass & Ceramics Products", "Textiles & Apparels", "Pulp & Paper", "Chemicals", "Pharmaceutical", 
              "Rubber Products", "Transportation Equipment", "Iron & Steel", "Nonferrous Metals", "Machinery", "Electric Appliances", "Precision Instruments", "Other Products", "Information & Communication", "Services", "Electric Power & Gas", 
              "Land Transportation", "Marine Transportation", "Air Transportation", "Warehousing & Harbor Transportation Services", "Wholesale Trade", "Retail Trade", "Banks", "Securities & Commodity Futures", "Insurance", "Other Financing Business", "Real Estate"]
    
    _REFINITIV_CODES = [".IFISH.T", ".IFOOD.T", ".IMING.T", ".IPETE.T", ".ICNST.T", ".IMETL.T", ".IGLSS.T", ".ITXTL.T", ".IPAPR.T", ".ICHEM.T", ".IPHAM.T", ".IRUBR.T", ".ITEQP.T", ".ISTEL.T", ".INFRO.T", ".IMCHN.T",
                        ".IELEC.T", ".IPRCS.T", ".IMISC.T", ".ICOMS.T", ".ISVCS.T", ".IEPNG.T", ".IRAIL.T", ".ISHIP.T", ".IAIRL.T", ".IWHSE.T", ".IWHOL.T", ".IRETL.T", ".IBNKS.T", ".ISECU.T", ".IINSU.T", ".IFINS.T", ".IRLTY.T"]
    
    _CODE_DICT = dict(zip(_NAMES, _REFINITIV_CODES))
    
    _prices = dict()

    def get_code(self, name: str) -> str:
        """ Helper method to return the Refinitiv TSE 33 code from the plain English language name.

        Returns:
            Refinitiv TSE 33 Code (str): Refinitiv PR Code like .IFISH.T or .IELEC.T
        """
        return self._CODE_DICT[name.replace(' and ', ' & ')]

    def get_prices(self, industry: str) -> pd.DataFrame:
        """ Helper method to return the TSI prices for a selected TSE 33 indsutry

        Returns:
            A TSE Sector's Prices (DataFrame): : Date, HIGH, LOW, OPEN, CLOSE, COUNT, VOLUME (empty)
        """
        industry_code = industry if industry.startswith('.I') else self.get_code(industry)
        idx = self._REFINITIV_CODES.index(industry_code)
        if not industry_code in self._prices.keys():
            self._prices[industry_code] = Security([idx, industry_code, self._NAMES[idx], 'TOPIX'])
        return self._prices[industry_code]



class Security:
    """Class to access minute interval security prices in data folder and add calculated return, 
    and calculated return adjusted for volatilty of entire history at each intraday minute

    Args:
        Securities Instrument (tuple): row number, ric, security name, optional TSE 33 Sector name (0, '2802.T', 'Ajinomoto Co Inc', 'Foods')
    """
    def _intraday_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        volat_df = pd.DataFrame(data=df[['TIME','RETURN']]).copy()
        volat_df['AVG_ABS_RETURN'] = volat_df.iloc[:,1].abs()
        volat_df['GK_VOL_RETURN'] = np.sqrt(((np.log(df['HIGH']) - np.log(df['LOW']))**2)*0.5 - ((2*np.log(2) - 1) * ((np.log(df['CLOSE']) - np.log(df['OPEN']))**2)))
        volat_df.GK_VOL_RETURN = np.where(volat_df.GK_VOL_RETURN.eq(0), volat_df.AVG_ABS_RETURN, volat_df.GK_VOL_RETURN)
        grouped = volat_df.groupby("TIME").mean()
        return grouped.reset_index()
    
    def _volatility_adjust_return(self, row):
        volatility = self.time_volatility[self.time_volatility['TIME'] == row['TIME']]['GK_VOL_RETURN'].item()
        return row['RETURN']/volatility

    def __init__(self, instrument: tuple):

        def parser(s):
            return datetime.strptime(s, '%Y-%m-%d %H:%M:%S')

        self.ric = instrument[1]
        self.name = instrument[2] 
        self.df = pd.read_csv(f"{DATA_PATH}/data/prices/minute_{self.ric}.csv", parse_dates=[0], date_parser=parser)
        self.df['DATE'] = pd.to_datetime(self.df['Date']).dt.date
        self.df['TIME'] = pd.to_datetime(self.df['Date']).dt.time
        self.df['RETURN'] = self.df['CLOSE'].pct_change(1)
        self.time_volatility = self._intraday_volatility(self.df)
        self.df['RETURN_ADJ'] = self.df.apply(self._volatility_adjust_return, axis = 1)

        


class Stock(Security):
    """Class to access minute interval stock prices in data folder and add calculated 
       RETURN  RETURN_ADJ  CLOSE_TPX  RETURN_TPX  RETURN_ADJ_TPX  RETURN_NORM  RETURN_NORM_TPX
       Interpolates some missing records in Stocks and Topix, calculate historical volatity adjustments and normalizaitons

    Args:
        Stock Instrument (tuple): row number, stock ric, stock name, TSE 33 Sector name (0, '2802.T', 'Ajinomoto Co Inc', 'Foods')
    """

    def _merge(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        m = df1.merge(df2.drop(['OPEN', 'HIGH', 'LOW', 'COUNT', 'VOLUME', 'DATE', 'TIME'], axis=1), on=['Date'], how='right', suffixes=(None, '_TPX'))
        m = m.sort_values(by=['Date']).reset_index(drop=True)
        start = m.CLOSE.first_valid_index()
        m = m[3:].reset_index(drop=True)
        for i in range(1,m.shape[0]):
            row = m.iloc[i].copy()
            if np.isnan(row['CLOSE']):
                previous_time = row["Date"] + pd.Timedelta(minutes=-1)
                prev_row = m.iloc[i-1]
                if prev_row['Date'] == previous_time and not np.isnan(prev_row['CLOSE']):
                    close = prev_row.CLOSE
                    row.CLOSE = close
                    row.HIGH = close
                    row.LOW = close
                    row.OPEN = close
                    row.COUNT, row.VOLUME = 0, 0
                    m.iloc[i] = row
        m['CLOSE_TPX'] = m['CLOSE_TPX'].interpolate()
        m.dropna(inplace=True)
        m = m.reset_index(drop=True)
        return m

    def __init__(self, instrument: tuple, TSE33: TOPIX_SECTORS):
        Security.__init__(self, instrument)
        #
        #  Stock prices missing some data
        #
        sector_name = instrument[3]
        self.sector = TSE33.get_prices(sector_name)
        self.df = self._merge(self.df, self.sector.df)
          
        # Date and row calculations
        self.unique_dates = np.sort(self.df['DATE'].unique())
        self.morning_start_idxs = [self.df.DATE.searchsorted(
            d, side='left') for d in self.unique_dates]
        self.afternoon_end_idxs = [self.df.DATE.searchsorted(
            d, side='right') - 1 for d in self.unique_dates]
        open_closes = self.morning_start_idxs.copy()
        open_closes.extend(self.afternoon_end_idxs)
        self.df_intraday = self.df.copy().drop(open_closes).reset_index(drop=True)
        # self.df_intraday['RETURN_NORM'] = self.df_intraday['RETURN_ADJ'].apply(
        #     lambda x: self.Scaler_return.learn_one({'x': x}).transform_one({'x': x})['x'])
        # self.df_intraday['RETURN_NORM_TPX'] = self.df_intraday['RETURN_ADJ_TPX'].apply(
        #         lambda x: self.Scaler_return_tpx.learn_one({'x': x}).transform_one({'x': x})['x'])
        self.df_intraday['RETURN_NORM'] = minmax(self.df_intraday['RETURN_ADJ'])
        self.df_intraday['RETURN_NORM_TPX'] = minmax(self.df_intraday['RETURN_ADJ_TPX'])


    def window(self, the_date: str, wdw_len: int = 5) ->pd.DataFrame:
        """Get price information for particular date

        Args:
            the_date (str): YYYY-mm-dd 
            wdw_len (int): number of days before the_date to include

        Returns:
            Stock price information (DataFrame): 
            Date HIGH LOW OPEN CLOSE COUNT VOLUME DATE TIME RETURN RETURN_ADJ CLOSE_TPX RETURN_TPX RETURN_ADJ_TPX RETURN_NORM RETURN_NORM_TPX
        """
        
        self.the_date = datetime.strptime(str(the_date), '%Y-%m-%d').date()
        unique_dates_idx_target = np.where(self.unique_dates == self.the_date)[0][0]
        unique_dates_wdw_start = unique_dates_idx_target - wdw_len
        self.wdw_start_date = self.unique_dates[unique_dates_wdw_start]
        self.the_date_start_idx = self.df_intraday.DATE.searchsorted(self.the_date, side='left')
        self.the_date_end_idx = self.df_intraday.DATE.searchsorted(self.the_date, side='right') - 1
        self.wdw_start_idx = self.df_intraday.DATE.searchsorted(self.wdw_start_date, side='left')
        self.wdw_end_idx = self.df_intraday.DATE.searchsorted(self.the_date, side='left') - 1
        unique_dates_wdw = self.unique_dates[unique_dates_wdw_start:unique_dates_idx_target]
        self.df_the_date = self.df_intraday.loc[self.wdw_start_idx:self.the_date_end_idx].copy().reset_index(drop=True)
        self.df_the_date['RETURN_NORM'] = minmax(self.df_the_date['RETURN_ADJ'])
        self.df_the_date['RETURN_NORM_TPX'] = minmax(self.df_the_date['RETURN_ADJ_TPX'])
        return self.df_the_date

class Nikkei225:
    """A helper class to get stock name and TSE 33 Industry Classification for a ric belonging to Nikkei 225"""

    def __init__(self):
        # be careful of spaces after commas in column names row
        self.df = pd.read_csv(f"{DATA_PATH}/data/Nikkei225.csv", header=0)
    
    def get_ric(self, ric: str) -> tuple:
        """Get information for Nikkei 225 ric.

        Args:
            ric (str): RIC Reuters/Refinitiv Instrument Code

        Returns:
            instrument (tuple): row number, stock ric, stock name, TSE 33 Sector name (0, '2802.T', 'Ajinomoto Co Inc', 'Foods')
        """
        return self.df[self.df['Instrument'] == ric].values[0]
    
    def get(self, idx: int):
        return self.df.loc[idx].values[0]

        
if __name__ == '__main__':
    TSE33 = TOPIX_SECTORS()
    NKY = Nikkei225()
    inst = NKY.get_ric("6758.T")
    print(f"inst: {inst}")
    s = Stock(inst, TSE33)
    sw = s.window("2021-03-09",1)
    print(sw.head())

