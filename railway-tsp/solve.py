import matplotlib.pyplot as plt
import time
from collections import defaultdict, deque


class Section:

    def __init__(self, from_station, to_station, from_time, to_time):
        self.from_station = from_station
        self.to_station = to_station
        self.from_time = from_time
        self.to_time = to_time

    def __repr__(self):
        return f'{self.from_station} ({self.from_time}) -> {self.to_station} ({self.to_time})'


def generate_sample_trains(
        n_stations=5,
        n_trains=10,
        first_time=300,
        train_interval=60,
        section_minutes=5):
    """
    テスト用の列車データを生成する関数

    Args:
        n_stations (int): 駅数
        n_trains (int): 列車本数
        first_time (int): 始発の出発時刻（0時からの経過分）
        train_interval (int): 各列車の出発間隔（分）
        section_minutes (int): 1区間あたりの所要時間（分）

    Returns:
        trains (list of list of `Section`): テスト用の列車データ
    """
    trains = list()
    for n in range(n_trains):
        base_time = first_time + n * train_interval
        direction = n % 2
        path = list()
        for from_station in range(n_stations - 1):
            to_station = from_station + 1
            from_time = base_time + from_station * section_minutes
            to_time = base_time + to_station * section_minutes
            if direction != 0:
                from_station, to_station = n_stations - 1 - from_station, n_stations - 1 - to_station
            path.append(Section(from_station, to_station, from_time, to_time))
        trains.append(path)
    return trains


def convert_to_timetable(trains):
    """
    列車データを時刻表データに変換する関数

    Args:
        trains (list of list of `Section`): 列車データ

    Returns:
        timetable (list): 時刻表データ
            timetable[from_station][to_station][dep_time] = (from_time, to_time)
            -> 現在時刻が dep_time の時に from_station から to_station まで直近の列車で移動する場合の
               乗車・下車時刻（0時からの経過分）のタプル
    """
    max_time = 1 + max([section.to_time for train in trains for section in train])
    n_stations = len(set([section.to_station for train in trains for section in train]))
    timetable = [[[(max_time, max_time) for _ in range(max_time)] for _ in range(n_stations)] for _ in range(n_stations)]
    # Step0: 次ステップの探索用に (時刻, 駅) についてのグラフ（adj）を作成
    adj = defaultdict(list)
    target_time_flag = [0 for _ in range(max_time)]
    for train in trains:
        for section in train:
            adj[(section.from_time, section.from_station)].append((section.to_time, section.to_station))
            target_time_flag[section.from_time] = 1
            target_time_flag[section.to_time] = 1
    target_times = [t for t in range(max_time) if target_time_flag[t] == 1]
    for station in range(n_stations):
        for from_time, to_time in zip(target_times[:-1], target_times[1:]):
            adj[(from_time, station)].append((to_time, station))
    # Step1: 出発時刻 = 乗車時刻 のデータを登録
    for train in trains:
        for section in train:
            # 他の駅への最速到着時刻をBFSで求める
            min_to_time = [max_time for _ in range(n_stations)]
            min_to_time[section.from_station] = section.from_time
            que = deque([(section.from_time, section.from_station)])
            visited = defaultdict(int)
            visited[(section.from_time, section.from_station)] = 1
            while len(que) > 0:
                from_time, from_station = que.popleft()
                for to_time, to_station in adj[(from_time, from_station)]:
                    if visited[(to_time, to_station)] == 1:
                        continue
                    min_to_time[to_station] = min(to_time, min_to_time[to_station])
                    que.append((to_time, to_station))
                    visited[(to_time, to_station)] = 1
            # 出発時刻 = 乗車時刻 のデータを登録
            for to_station in range(n_stations):
                if to_station == section.from_station:
                    continue
                to_time = min_to_time[to_station]
                if to_time == max_time:
                    continue
                timetable[section.from_station][to_station][section.from_time] = (section.from_time, to_time)
    # Step2: 出発時刻 != 乗車時刻 のデータを登録
    #     例えば駅1→2の始発列車を考え、5:00（300）発・5:05（305）着だとする。
    #     step1では timetable[1][2][300] = (300, 305) とデータが登録される。
    #     ここで駅1を5:00(300)より前に出発するとしても、駅1で待機して同じ列車に乗ることになるため、
    #     t < 300 に対して timetable[1][2][t] = (300, 305) となるはず。
    #     step1ではこのデータは入らないので、ここで入れる。
    for t in range(max_time - 2, - 1, - 1):
        for from_station in range(n_stations):
            for to_station in range(n_stations):
                timetable[from_station][to_station][t] = \
                   min(timetable[from_station][to_station][t], timetable[from_station][to_station][t + 1])
    return timetable


def find_optimal_route_by_bit_dp(timetable, start_station=0, stay_minutes=10):
    """
    bit dpで最適ルートを求める関数

    Args:
        timetable (list): 時刻表データ（generate_sample_timetable で生成される形式）
        start_station (int): 移動開始の駅（＝移動の最終目的駅）
        stay_minutes (int): 各駅における滞在時間（分）

    Returns:
        optimal_path (list of `Section`): 最適ルート
    """
    n_stations = len(timetable)
    max_time = len(timetable[0][0])
    # dp[n][s]: 「駅集合s内の駅に全て訪問済み＆最後に訪問したのが駅n」の場合の最も早い到着時刻
    dp = [[max_time for _ in range(1 << n_stations)] for _ in range(n_stations)]
    # dpの各状態における親の状態（経路復元に利用）
    parent = [[(None, None) for _ in range(1 << n_stations)] for _ in range(n_stations)]
    # bit dp
    dp[start_station][0] = 0
    for state in range(1, 1 << n_stations):
        for to_station in range(n_stations):
            # to_station: stateの中で最後に訪れた駅とする（なので未訪問の場合はスルー）
            if (state >> to_station) & 1 == 0:
                continue
            min_to_time = max_time
            min_parent = (None, None)
            from_state = state - (1 << to_station)
            for from_station in range(n_stations):
                if from_station == to_station:
                    continue
                if from_state != 0 and (from_state >> from_station) & 1 == 0:
                    continue
                current_time = dp[from_station][from_state]
                current_time += stay_minutes
                if current_time >= max_time:
                    continue
                _, to_time = timetable[from_station][to_station][current_time]
                if to_time < min_to_time:
                    min_to_time = to_time
                    min_parent = (from_station, from_state)
            dp[to_station][state] = min_to_time
            parent[to_station][state] = min_parent
    # 経路復元
    optimal_path = list()
    to_station, to_state = start_station, (1 << n_stations) - 1
    from_station, from_state = parent[to_station][to_state]
    while from_station is not None:
        current_time = dp[from_station][from_state] + stay_minutes
        from_time, to_time = timetable[from_station][to_station][current_time]
        optimal_path.append(Section(from_station, to_station, from_time, to_time))
        to_station, to_state = from_station, from_state
        from_station, from_state = parent[to_station][to_state]
    optimal_path.reverse()
    return optimal_path


def draw_diagram(trains, path=[]):
    # 列車を図示
    for train in trains:
        for section in train:
            plt.plot([section.from_time, section.to_time], [section.from_station, section.to_station],
                     color='g', marker='o', markersize=3)
        for section, next_section in zip(train[:-1], train[1:]):
            plt.plot([section.to_time, next_section.from_time], [section.to_station, next_section.from_station],
                     color='g', marker='o', markersize=3)
    # （もしあれば）移動経路を重ねて図示
    if len(path) > 0:
        for section in path:
            plt.plot([section.from_time, section.to_time], [section.from_station, section.to_station],
                     color='r', linewidth=3)
        for section, next_section in zip(path[:-1], path[1:]):
            plt.plot([section.to_time, next_section.from_time], [section.to_station, next_section.from_station],
                     color='r', linewidth=3)
    plt.xlabel('time')
    plt.ylabel('station')
    plt.show()


def main():
    # *** テストケース1：人工のケース（小さめのテストケース） ***
    print('*** Test case 1 ***')
    trains = generate_sample_trains(n_stations=10, n_trains=18, train_interval=55)
    start_time = time.time()
    timetable = convert_to_timetable(trains)
    conversion_time = time.time()
    print('conversion time: {} sec'.format(conversion_time - start_time))
    optimal_path = find_optimal_route_by_bit_dp(timetable, stay_minutes=30)
    end_time = time.time()
    print('search time: {} sec'.format(end_time - conversion_time))
    draw_diagram(trains, optimal_path)
    # *** テストケース 2：人工のケース（やや大きめのテストケース） ***
    print('*** Test case 2 ***')
    trains = generate_sample_trains(n_stations=17, n_trains=30, train_interval=40)
    start_time = time.time()
    timetable = convert_to_timetable(trains)
    conversion_time = time.time()
    print('conversion time: {} sec'.format(conversion_time - start_time))
    optimal_path = find_optimal_route_by_bit_dp(timetable, stay_minutes=15)
    end_time = time.time()
    print('search time: {} sec'.format(end_time - conversion_time))
    draw_diagram(trains, optimal_path)
    # *** テストケース 3：実例（わたらせ渓谷鉄道、桐生から出発して桐生に戻る） ***
    print('*** Test case 3 ***')
    trains = generate_watarase_trains()
    start_time = time.time()
    timetable = convert_to_timetable(trains)
    conversion_time = time.time()
    print('conversion time: {} sec'.format(conversion_time - start_time))
    optimal_path = find_optimal_route_by_bit_dp(timetable, stay_minutes=15)
    end_time = time.time()
    print('search time: {} sec'.format(end_time - start_time))
    draw_diagram(trains, optimal_path)


def generate_watarase_trains():
    """
    Returns:
        trains (list of list of `Section`): わたらせ渓谷鉄道の列車データ(2020年冬)
            https://www.watetsu.com/jikoku_torokko/201201_210331.pdf
            ・駅のインデックスは「桐生:0、…、間藤：16」とした
            ・臨時列車及び日光市営バスは利用しない前提
    """
    # 1セクションの発着情報を発着交互に並べたもの
    data = [
        # 間藤方面
        {
            'station': [15,16],
            'time': [539,542]
        },
        {
            'station': [15,16],
            'time': [622,625]
        },
        {
            'station': [0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16],
            'time': [636,640,640,642,643,645,645,650,658,706,706,708,708,716,716,723,723,725,725,729,729,733,734,746,746,755,755,800,800,803,803,806]
        },
        {
            'station': [0,1,1,2,2,3,3,4],
            'time': [724,727,727,730,732,734,734,739]
        },
        {
            'station': [0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16],
            'time': [744,747,747,750,753,755,755,800,804,811,811,814,814,820,820,826,826,828,828,832,832,836,841,852,852,902,902,907,907,910,910,913]
        },
        {
            'station': [0,1,1,2,2,3,3,4],
            'time': [812,815,815,818,818,820,820,825]
        },
        {
            'station': [0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16],
            'time': [854,858,858,900,905,907,907,912,916,924,924,927,927,935,935,941,941,943,943,947,947,951,955,1006,1006,1016,1016,1021,1021,1024,1024,1027]
        },
        {
            'station': [0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16],
            'time': [1005,1008,1008,1011,1015,1017,1017,1022,1027,1034,1034,1037,1037,1044,1044,1050,1050,1052,1052,1056,1056,1100,1105,1120,1120,1129,1129,1135,1135,1137,1137,1141]
        },
        {
            'station': [0,1,1,2,2,3,3,4],
            'time': [1059,1102,1102,1105,1111,1113,1113,1118]
        },
        {
            'station': [0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16],
            'time': [1127,1130,1130,1133,1136,1138,1138,1143,1146,1153,1153,1156,1156,1203,1203,1209,1209,1211,1211,1215,1215,1219,1229,1244,1244,1253,1253,1259,1259,1301,1301,1305]
        },
        {
            'station': [0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16],
            'time': [1306,1309,1309,1312,1320,1322,1322,1327,1330,1337,1337,1340,1340,1346,1346,1352,1352,1354,1354,1358,1358,1402,1408,1424,1424,1433,1433,1439,1439,1441,1441,1445]
        },
        {
            'station': [0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16],
            'time': [1435,1439,1439,1441,1445,1448,1448,1452,1457,1504,1504,1507,1507,1517,1517,1523,1523,1525,1525,1529,1529,1533,1540,1556,1556,1605,1605,1611,1611,1613,1613,1617]
        },
        {
            'station': [0,1,1,2,2,3,3,4],
            'time': [1513,1516,1516,1519,1519,1521,1521,1526]
        },
        {
            'station': [0,1,1,2,2,3,3,4],
            'time': [1617,1621,1621,1623,1628,1630,1630,1635]
        },
        {
            'station': [0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16],
            'time': [1646,1649,1649,1652,1655,1657,1657,1702,1705,1712,1712,1715,1715,1721,1721,1727,1727,1729,1729,1733,1733,1737,1738,1750,1750,1759,1759,1805,1805,1807,1807,1811]
        },
        {
            'station': [0,1,1,2,2,3,3,4],
            'time': [1727,1730,1730,1733,1733,1735,1735,1740]
        },
        {
            'station': [0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16],
            'time': [1804,1807,1807,1810,1810,1812,1812,1817,1820,1827,1827,1830,1830,1836,1836,1842,1842,1844,1844,1848,1848,1852,1853,1905,1905,1914,1914,1920,1920,1922,1922,1926]
        },
        {
            'station': [0,1,1,2,2,3,3,4],
            'time': [1850,1854,1854,1856,1857,1859,1859,1904]
        },
        {
            'station': [0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15,16],
            'time': [1956,2000,2000,2002,2003,2005,2005,2010,2013,2020,2020,2023,2023,2031,2031,2037,2037,2039,2039,2043,2043,2047,2048,2100,2100,2109,2109,2115,2115,2117,2117,2121]
        },
        {
            'station': [0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13,14,14,15],
            'time': [2125,2128,2128,2131,2131,2133,2133,2138,2139,2147,2147,2150,2150,2156,2156,2202,2202,2204,2204,2208,2208,2212,2213,2224,2224,2234,2234,2239,2239,2241]
        },
        # 桐生方面
        {
            'station': [4,3,3,2,2,1,1,0],
            'time': [604,608,608,610,611,613,613,617]
        },
        {
            'station': [16,15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8,7,7,6,6,5,5,4,4,3,3,2,2,1,1,0],
            'time': [547,550,550,552,552,557,557,606,606,614,615,619,619,623,623,625,625,632,632,637,637,640,640,647,650,655,655,657,700,702,702,706]
        },
        {
            'station': [4,3,3,2,2,1,1,0],
            'time': [720,725,725,727,732,734,734,737]
        },
        {
            'station': [16,15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8,7,7,6,6,5,5,4,4,3,3,2,2,1,1,0],
            'time': [630,633,633,636,636,641,641,650,650,658,659,703,703,708,708,710,710,720,720,725,725,728,728,736,745,750,750,752,754,756,756,800]
        },
        {
            'station': [4,3,3,2,2,1,1,0],
            'time': [808,813,813,815,818,820,820,824]
        },
        {
            'station': [16,15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8,7,7,6,6,5,5,4,4,3,3,2,2,1,1,0],
            'time': [812,815,815,817,817,823,823,832,832,840,843,847,847,851,851,853,853,900,900,905,905,908,908,916,924,928,928,930,937,939,939,942]
        },
        {
            'station': [16,15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8,7,7,6,6,5,5,4,4,3,3,2,2,1,1,0],
            'time': [920,923,923,925,925,931,931,940,940,948,952,956,956,1000,1000,1002,1002,1009,1009,1014,1014,1017,1017,1025,1027,1031,1031,1033,1038,1041,1041,1044]
        },
        {
            'station': [4,3,3,2,2,1,1,0],
            'time': [1104,1108,1108,1110,1111,1114,1114,1117]
        },
        {
            'station': [16,15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8,7,7,6,6,5,5,4,4,3,3,2,2,1,1,0],
            'time': [1058,1104,1104,1106,1106,1111,1111,1120,1120,1129,1140,1143,1143,1148,1148,1150,1150,1202,1202,1207,1207,1210,1210,1218,1222,1226,1226,1228,1235,1238,1238,1241]
        },
        {
            'station': [16,15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8,7,7,6,6,5,5,4,4,3,3,2,2,1,1,0],
            'time': [1222,1227,1227,1230,1230,1235,1235,1244,1244,1252,1255,1259,1259,1303,1303,1305,1305,1314,1314,1319,1319,1321,1321,1329,1331,1336,1336,1338,1342,1344,1344,1348]
        },
        {
            'station': [16,15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8,7,7,6,6,5,5,4,4,3,3,2,2,1,1,0],
            'time': [1328,1331,1331,1333,1333,1338,1338,1347,1347,1356,1403,1406,1406,1411,1411,1413,1413,1420,1420,1425,1425,1427,1427,1435,1438,1443,1443,1445,1448,1450,1450,1454]
        },
        {
            'station': [4,3,3,2,2,1,1,0],
            'time': [1546,1550,1550,1552,1600,1602,1602,1606]
        },
        {
            'station': [16,15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8,7,7,6,6,5,5,4,4,3,3,2,2,1,1,0],
            'time': [1509,1515,1515,1517,1517,1522,1522,1531,1531,1539,1540,1544,1544,1548,1548,1550,1550,1557,1557,1602,1602,1605,1605,1613,1619,1624,1624,1626,1630,1633,1633,1636]
        },
        {
            'station': [4,3,3,2,2,1,1,0],
            'time': [1705,1710,1710,1712,1716,1718,1718,1722]
        },
        {
            'station': [16,15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8,7,7,6,6,5,5,4,4,3,3,2,2,1,1,0],
            'time': [1631,1634,1634,1636,1636,1642,1642,1651,1651,1659,1701,1705,1705,1709,1709,1712,1712,1721,1721,1726,1726,1728,1728,1736,1741,1746,1746,1748,1751,1753,1753,1757]
        },
        {
            'station': [4,3,3,2,2,1,1,0],
            'time': [1825,1830,1830,1832,1835,1837,1837,1841]
        },
        {
            'station': [16,15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8,7,7,6,6,5,5,4,4,3,3,2,2,1,1,0],
            'time': [1821,1824,1824,1826,1826,1831,1831,1840,1840,1849,1854,1857,1857,1902,1902,1904,1904,1911,1911,1916,1916,1918,1918,1926,1930,1935,1935,1937,1941,1943,1943,1947]
        },
        {
            'station': [16,15,15,14,14,13,13,12,12,11,11,10,10,9,9,8,8,7,7,6,6,5,5,4,4,3,3,2,2,1,1,0],
            'time': [1946,1949,1949,1951,1951,1956,1956,2005,2005,2014,2015,2018,2018,2023,2023,2025,2025,2032,2032,2037,2037,2039,2039,2047,2048,2053,2053,2055,2056,2059,2059,2102]
        },
        {
            'station': [16,15],
            'time': [2126,2128]
        }
    ]
    # Sectionのlistのlistに変換
    trains = list()
    for row in data:
        n = len(row['station'])
        assert n == len(row['time'])
        assert n % 2 == 0
        train = list()
        for i in range(n // 2):
            from_station = row['station'][2 * i]
            to_station = row['station'][2 * i + 1]
            from_time = 60 * (row['time'][2 * i] // 100) + row['time'][2 * i] % 100
            to_time = 60 * (row['time'][2 * i + 1] // 100) + row['time'][2 * i + 1] % 100
            train.append(Section(from_station, to_station, from_time, to_time))
        trains.append(train)
    return trains


if __name__ == '__main__':
    main()
