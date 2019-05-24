from collections import deque
import numpy as np

class RecognizeResult:
    def __init__(self):
        self.found_command = '_silence_'
        self.score = 0
        self.is_new_command = False

class RecognizeCommands:
    def __init__(self, labels, average_window_duration_ms, detection_threshold, suppression_ms, minimum_count):
        # Configuration
        self._labels = labels
        self._average_window_duration_ms = average_window_duration_ms
        self._detection_threshold = detection_threshold
        self._suppression_ms = suppression_ms
        self._minimum_count = minimum_count

        # Working Variable
        self._previous_results = deque()
        self._label_count = len(labels)
        self._previous_top_label = '_silence_'
        self._previous_top_label_time = -np.inf

    def process_latest_results(self, latest_results, current_time_ms, recognize_element):
        if latest_results.shape[0] != self._label_count:
            raise ValueError("The results for recognition should contain ", self._label_count,
                             " elements, but there are ", latest_results.shape[0])

        if self._previous_results.__len__() != 0 and current_time_ms < self._previous_results[0][0]:
            raise ValueError("Results must be fed in increasing time order, but receive a "
                             "timestamp of ",
                             current_time_ms, " that was earlier than the previous one of ",
                             self._previous_results[0][0])

        # Add the latest result to the head of the deque.
        self._previous_results.append([current_time_ms, latest_results])

        # Prune any earlier results that are too old for the averaging window.
        time_limit = current_time_ms - self._average_window_duration_ms
        while time_limit > self._previous_results[0][0]:
            self._previous_results.popleft()

        # If there are to few results, assume the result will be unreliable and bail.
        how_many_results = self._previous_results.__len__()
        earliest_time = self._previous_results[0][0]
        sample_duration = current_time_ms - earliest_time
        if how_many_results < self._minimum_count or sample_duration < self._average_window_duration_ms / 4:
            recognize_element.found_command = self._previous_top_label
            recognize_element.score = 0.0
            recognize_element.is_new_command = False
            return

        # Calculate the average score across all the results in the window.
        average_scores = np.zeros(self._label_count)
        for item in self._previous_results:
            score = item[1]
            for i in range(score.size):
                average_scores[i] += score[i] / how_many_results

        # Sort the averaged results in descending score order.
        sorted_averaged_index_score = []
        for i in range(self._label_count):
            sorted_averaged_index_score.append([i, average_scores[i]])
        sorted_averaged_index_score = sorted(sorted_averaged_index_score, key=lambda p: p[1], reverse=True)

        current_top_index = sorted_averaged_index_score[0][0]
        current_top_label = self._labels[current_top_index]
        current_top_score = sorted_averaged_index_score[0][1]

        time_since_last_top = 0

        if self._previous_top_label == '_silence_' or self._previous_top_label_time == -np.inf:
            time_since_last_top = np.inf
        else:
            time_since_last_top = current_time_ms - self._previous_top_label_time

        if current_top_score > self._detection_threshold and \
                current_top_label != self._previous_top_label and \
                time_since_last_top > self._suppression_ms:
            self._previous_top_label = current_top_label
            self._previous_top_label_time = current_time_ms
            recognize_element.is_new_command = True
        else:
            recognize_element.is_new_command = False
        recognize_element.found_command = current_top_label
        recognize_element.score = current_top_score
