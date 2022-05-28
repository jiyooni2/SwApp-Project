from range_selector import RangeSelector
import pandas as pd
import csv
from sqlalchemy import create_engine
import pymysql
import os.path
import matplotlib.pyplot as plt
import seaborn as sns

pymysql.install_as_MySQLdb()

engine = create_engine("mysql+mysqldb://root@localhost/school", encoding='utf-8')
conn = engine.connect()
apt_combined_df = None

if os.path.isfile("combined_df.csv"):
    apt_combined_df = pd.read_csv("combined_df.csv")
    print("Loaded pre-calculated data.")
else:
    # Initialize environments
    selector = RangeSelector(1.0, "processed.json")
    print("Loaded RangeSelector.")

    apt_coord = {}
    with open("data/coordlist.csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        for record in csv_reader:
            apt_coord[record[0]] = { "x": float(record[1]), "y": float(record[2]) }

    apt_combined = {
        **({key: [] for key in selector.datatypes}),
        **({(key+"-dist"): [] for key in selector.datatypes})
    }
    apt_combined["price"] = []
    apt_combined["name"] = []

    apt_combined["x"] = []
    apt_combined["y"] = []

    # Parse price table
    sql = "select * from price"
    price = pd.read_sql_query(sql, conn)
    price["이름"] = price['동'] + " " + price['아파트']
    price["전용면적"] = price["전용면적"].apply(lambda record: round(record / 3.3))
    price = price.drop_duplicates(subset=["이름"], keep="last")

    print("Loaded apt data from price table.")

    matched = 0; iter_count = 0
    for price_row in price.itertuples():
        iter_count += 1
        print(f"Processing... ({round(iter_count/len(price.index) * 100, 3)} %)  ", end="\r")
        
        if price_row.이름 in apt_coord:
            name = price_row.이름
            apt_combined["price"].append(price_row.금액)
            apt_combined["name"].append(price_row.이름)
            apt_combined["x"].append(apt_coord[name]["x"])
            apt_combined["y"].append(apt_coord[name]["y"])
            count = selector.count_neibors(apt_coord[name]["x"], apt_coord[name]["y"])
            for datatype in count:
                apt_combined[datatype].append(count[datatype])
            
            dist = selector.find_shortest(apt_coord[name]["x"], apt_coord[name]["y"])
            for datatype in dist:
                apt_combined[datatype+"-dist"].append(dist[datatype])
            
            matched += 1
    print(f"\n{matched / iter_count * 100}% is matched.")

    apt_combined_df = pd.DataFrame(apt_combined)
    print(apt_combined_df)

    # Parse aptinfo table
    sql = "select * from aptinfo"
    aptinfo = pd.read_sql_query(sql, conn)
    apt_combined_df = pd.merge(left=apt_combined_df, right=aptinfo, how="inner", left_on="name", right_on="아파트")
    apt_combined_df = apt_combined_df.drop("아파트", axis=1)

    apt_combined_df.to_csv("./combined_df.csv", ",")
    print("Saved data.")

# # 대치동만 포함
# apt_combined_df = apt_combined_df[apt_combined_df["dong"].str.contains("대치동")]

print("-- Sample --")
print(apt_combined_df.loc[[0]])

# 위치 정보 제외
apt_combined_df = apt_combined_df.drop(["x", "y"], axis=1)

corr_matrix = apt_combined_df.corr()
fig, ax = plt.subplots(figsize=(32,24))
sns.heatmap(corr_matrix, annot=True, fmt=".4f",
            cmap=plt.get_cmap('coolwarm'), cbar=False, ax=ax)
ax.set_yticklabels(ax.get_yticklabels(), rotation='horizontal')
plt.savefig("corr.png", bbox_inches='tight', pad_inches=0.0, dpi=500)

fig, ax = plt.subplots()
ax.scatter(apt_combined_df["academy"], apt_combined_df["price"], c=apt_combined_df["price"])
plt.savefig("academy_scatter.png")