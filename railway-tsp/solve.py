from itertools import permutations
import matplotlib.pyplot as plt


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
    テスト用の時刻表データを生成する関数

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
    # Step1: 出発時刻 = 乗車時刻 のデータを登録
    for train in trains:
        for i, first_section in enumerate(train):
            for last_section in train[i:]:
                timetable[first_section.from_station][last_section.to_station][first_section.from_time] = \
                    (first_section.from_time, last_section.to_time)
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


def find_optimal_route_by_full_search(timetable, start_station=0, stay_minutes=10):
    """
    全探索で最適ルートを求める関数

    Args:
        timetable (list): 時刻表データ（generate_sample_timetable で生成される形式）
        start_station (int): 移動開始の駅（＝移動の最終目的駅）
        stay_minutes (int): 各駅における滞在時間（分）

    Returns:
        optimal_path (list of `Section`): 最適ルート
    """
    n_stations = len(timetable)
    max_time = len(timetable[0][0])
    min_time, optimal_path = max_time, None
    for stations in permutations([i for i in range(n_stations) if i != start_station], n_stations - 1):
        stations = list(stations)
        stations.append(start_station)  # 最後に start_station に戻る
        from_station, current_time = start_station, 0
        path = list()
        for to_station in stations:
            if current_time > max_time:
                break
            from_time, to_time = timetable[from_station][to_station][current_time]
            path.append(Section(from_station, to_station, from_time, to_time))
            current_time = to_time
            if to_station != start_station:
                current_time += stay_minutes
            from_station = to_station
        if min_time > current_time:
            min_time = current_time
            optimal_path = path
    return optimal_path


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
        for section, next_section in zip(train[1:], train[:-1]):
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
    import time
    # trains = generate_sample_trains(n_stations=16, n_trains=26, train_interval=40)
    trains = generate_sample_trains(n_stations=9)
    timetable = convert_to_timetable(trains)
    # (1) Full Search
    start_time = time.time()
    optimal_path = find_optimal_route_by_full_search(timetable)
    end_time = time.time()
    print('full search: {} sec'.format(end_time - start_time))
    draw_diagram(trains, optimal_path)
    # (2) Bit DP
    start_time = time.time()
    optimal_path = find_optimal_route_by_bit_dp(timetable, stay_minutes=15)  # 5 or 15
    end_time = time.time()
    print('bit dp: {} sec'.format(end_time - start_time))
    draw_diagram(trains, optimal_path)


if __name__ == '__main__':
    main()
