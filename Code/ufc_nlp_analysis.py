import sqlite3
import pandas as pd

data_dir = '../Data/'

conn = sqlite3.connect(data_dir + 'db.sqlite3')

# cursor = conn.cursor()
# cursor.execute('''SELECT * FROM fights;''')
# data = cursor.fetchall()
# cursor.close()

# read data into pandas df
sql_query = '''SELECT * FROM fights;'''
data_df = pd.read_sql_query(sql_query, conn)

# summary stats
