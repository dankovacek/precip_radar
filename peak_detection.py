import numpy as np

# Let y be a vector of timeseries data of at least length lag+2
# Let mean() be a function that calculates the mean
# Let std() be a function that calculates the standard deviaton
# Let absolute() be the absolute value function

def find_peaks(data, lag=5, threshold=3.5, influence=0.5):
    # Settings (the ones below are examples: choose what is best for your data)
#     lag = 5         # lag 5 for the smoothing functions
#     threshold = 3.5  # 3.5 standard deviations for signal
#     influence = 0.5  # between 0 and 1, where 1 is normal influence, 0.5 is half
    # Initialize variables
    signals = np.zeros(len(data))             # Initialize signal results
    filteredY = data[:lag]                    # Initialize filtered series
    avgFilter = None                          # Initialize average filter
    stdFilter = None                          # Initialize std. filter
    avgFilter = {lag: np.mean(data[:lag])}      # Initialize first value
    stdFilter = {lag: np.std(data[:lag])}     # Initialize first value

    for i in range(lag + 1, len(data)):
        
        d = data[i]
        print(i, d)
        if abs(d - avgFilter[i-1]) > threshold * stdFilter[i-1]:
            if d > avgFilter[i-1]:
                signals[i] = 1                     # Positive signal
            else:
                signals[i] = -1                      # Negative signal

        # Reduce influence of signal
            filteredY[i] = influence*d + (1-influence)*filteredY[i-1];
        else:
            signals[i] = 0                        # No signal
            filteredY[i] = d

      # Adjust the filters
        avgFilter[i] = np.mean(filteredY[i-lag:i])
        stdFilter[i] = np.std(filteredY[i-lag:i])


# class real_time_peak_detection():
#     """
#     Robust peak signal detection:
#     https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/56451135#56451135
#     """
#     def __init__(self, array, lag, threshold, influence):
#         self.y = list(array)
#         self.length = len(self.y)
#         self.lag = lag
#         self.threshold = threshold
#         self.influence = influence
#         self.signals = [0] * len(self.y)
#         self.filteredY = np.array(self.y).tolist()
#         self.avgFilter = [0] * len(self.y)
#         self.stdFilter = [0] * len(self.y)
#         self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag]).tolist()
#         self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag]).tolist()

#     def thresholding_algo(self, new_value):
#         self.y.append(new_value)
#         i = len(self.y) - 1
#         self.length = len(self.y)
#         if i < self.lag:
#             return 0
#         elif i == self.lag:
#             self.signals = [0] * len(self.y)
#             self.filteredY = np.array(self.y).tolist()
#             self.avgFilter = [0] * len(self.y)
#             self.stdFilter = [0] * len(self.y)
#             self.avgFilter[self.lag - 1] = np.mean(self.y[0:self.lag]).tolist()
#             self.stdFilter[self.lag - 1] = np.std(self.y[0:self.lag]).tolist()
#             return 0

#         self.signals += [0]
#         self.filteredY += [0]
#         self.avgFilter += [0]
#         self.stdFilter += [0]

#         if abs(self.y[i] - self.avgFilter[i - 1]) > self.threshold * self.stdFilter[i - 1]:
#             if self.y[i] > self.avgFilter[i - 1]:
#                 self.signals[i] = 1
#             else:
#                 self.signals[i] = -1

#             self.filteredY[i] = self.influence * self.y[i] + (1 - self.influence) * self.filteredY[i - 1]
#             self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
#             self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])
#         else:
#             self.signals[i] = 0
#             self.filteredY[i] = self.y[i]
#             self.avgFilter[i] = np.mean(self.filteredY[(i - self.lag):i])
#             self.stdFilter[i] = np.std(self.filteredY[(i - self.lag):i])

#         return self.signals[i]
