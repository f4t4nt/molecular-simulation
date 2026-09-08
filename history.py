import pandas as pd


def get_df_posHistoryArr(positionHistoryArr, df):
  df2 = pd.DataFrame(data = positionHistoryArr, columns=["time", "atomId", "posX", "posY", "posZ"]) \
    .astype({"atomId": "int16"})

  if df is None:
    return df2

  return pd.concat([df, df2])

def get_df_tickHistoryArr(tickHistoryArray, df):
  tickHistDf = pd.DataFrame(
    data = tickHistoryArray,
    columns = ["time", "potentialE", "kineticE", "CC_Bonds", "CH_Bonds"])

  if df is None:
    return tickHistDf

  return pd.concat([df, tickHistDf])
