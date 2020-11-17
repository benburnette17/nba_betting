# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import gambling
import ou_classifier


def bet_ben(amount = 1000, odds = 48.5/50):


    d = ou_classifier.OU_predictor(model_type="custom", model_param=10)
    data = d.simDict()
    starting = amount
    current = amount

    for i in range(len(data)):
        game_info = data[i]
        profit = current - starting
        if profit > 0:
            bet = 75 + .4 * profit
        elif current < 75:
            bet = current
        else:
            bet = 75
        if game_info[1] > .7 and game_info[0] == game_info[2]:
            current += bet * odds
        elif game_info[1] > .7 and game_info[0] != game_info[2]:
            current -= bet
        else:
            continue
        if current <= 0:
            #print("Welp, you lost all your money. Don't gamble kids.")
            return 0

    #print("Ben's amount of money at the end is: ", current)
    return current

def bet_eli(amount = 1000, odds = 48.5/50):


    d = ou_classifier.OU_predictor(model_type="custom", model_param=10)
    data = d.simDict()
    current = amount

    for i in range(len(data)):
        game_info = data[i]
        bet = 20
        if game_info[1] > .8 and game_info[0] == game_info[2]:
            current += bet * odds
        elif game_info[1] > .8 and game_info[0] != game_info[2]:
            current -= bet
        else:
            continue
        if current <= 0:
            #print("Welp, you lost all your money. Don't gamble kids.")
            return 0

    #print("Eli's amount of money at the end is: ", current)
    return current

def bet_daniel(amount = 1000, odds = 48.5/50):


    d = ou_classifier.OU_predictor(model_type="custom", model_param=10)
    data = d.simDict()
    current = amount

    for i in range(len(data)):
        game_info = data[i]
        bet = 100
        if game_info[1] > .8 and game_info[0] == game_info[2]:
            current += bet * odds
        elif game_info[1] > .8 and game_info[0] != game_info[2]:
            current -= bet
        else:
            continue
        if current <= 0:
            #print("Welp, you lost all your money. Don't gamble kids.")
            return 0

    #print("Daniel's amount of money at the end is: ", current)
    return current

def simulate_bets():
    print("Running simulation for Ben")
    ben_max = float("-inf")
    ben_min = float("inf")
    ben_mean = 0
    ben_array = []
    for i in range(100):
        simulation = bet_ben()
        if simulation > ben_max:
            ben_max = simulation
        if simulation < ben_min:
            ben_min = simulation
        ben_mean += simulation
        ben_array.append(simulation)
    ben_mean /= 100
    ben_array.sort()

    print("Ben's max: ", ben_max)
    print("Ben's min: ", ben_min)
    print("Ben's avg: ", ben_mean)
    print("Ben's median: ", ben_array[50])

    print()

    print("Running simulation for Eli")
    eli_max = float("-inf")
    eli_min = float("inf")
    eli_mean = 0
    eli_array = []
    for i in range(100):
        simulation = bet_eli()
        if simulation > eli_max:
            eli_max = simulation
        if simulation < eli_min:
            eli_min = simulation
        eli_mean += simulation
        eli_array.append(simulation)
    eli_mean /= 100
    eli_array.sort()
    print("Eli's max: ", eli_max)
    print("Eli's min: ", eli_min)
    print("Eli's avg: ", eli_mean)
    print("Eli's median: ", eli_array[50])


    print()

    print("Running simulation for Daniel")
    dan_max = float("-inf")
    dan_min = float("inf")
    dan_mean = 0
    dan_array = []
    for i in range(100):
        simulation = bet_daniel()
        if simulation > dan_max:
            dan_max = simulation
        if simulation < dan_min:
            dan_min = simulation
        dan_mean += simulation
        dan_array.append(simulation)
    dan_mean /= 100
    dan_array.sort()

    print("Daniel's max: ", dan_max)
    print("Daneil's min: ", dan_min)
    print("Daniel's avg: ", dan_mean)
    print("Daniel's median: ", dan_array[50])



if __name__ == '__main__':
    simulate_bets()