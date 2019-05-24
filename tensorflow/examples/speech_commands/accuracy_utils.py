import numpy as np
import logging


class StreamingAccuracyStats:
    def __init__(self):
        self._how_many_ground_truth_words = 0
        self._how_many_ground_truth_matched = 0
        self._how_many_false_positives = 0
        self._how_many_correct_words = 0
        self._how_many_wrong_words = 0
        self._ground_truth_occurence = []

        self._previous_correct_words = 0
        self._previous_wrong_words = 0
        self._previous_false_positives = 0

    # Takes a file name, and loads a list of expected word labels and items
    # from it, as  comma-separated variables.
    def read_ground_truth_file(self, file_name):

        with open(file_name, 'r') as f:
            for line in f.readlines():
                line_split = line.strip().split(',')
                if len(line_split) != 2:
                    continue
                timestamp = round(float(line_split[1]))
                label = line_split[0]
                self._ground_truth_occurence.append([label, timestamp])

        self._ground_truth_occurence = sorted(self._ground_truth_occurence,
                                              key=lambda item: item[1])

    # Compute delta of statistic
    def delta(self):
        false_positives_delta = self._how_many_false_positives - self._previous_false_positives
        wrong_delta = self._how_many_wrong_words - self._previous_wrong_words
        correct_delta = self._how_many_correct_words - self._previous_correct_words
        if false_positives_delta == 1:
            recognition_state = " (False Positive)"
        elif correct_delta == 1:
            recognition_state = " (Correct)"
        elif wrong_delta == 1:
            recognition_state = " (Wrong)"
        else:
            raise ValueError("Unexpected state in statistics")

        self._previous_correct_words = self._how_many_correct_words
        self._previous_wrong_words = self._how_many_wrong_words
        self._previous_false_positives = self._how_many_false_positives

        return recognition_state

    # Given ground truth labels and corresponding predictions found by a model,
    # figure out how many were correct. Takes a time limit, so that only
    # predictions up to a point in time are considered, in case we're evaluating
    # accuracy when the model has only been run on part of the stream.
    def calculate_accuracy_stats(self, found_words, up_to_time_ms, time_tolerance_ms):
        if up_to_time_ms == -1:
            latest_possible_time = np.inf
        else:
            latest_possible_time = up_to_time_ms + time_tolerance_ms
        self._how_many_ground_truth_words = 0
        for ground_truth in self._ground_truth_occurence:
            ground_truth_time = ground_truth[1]
            if ground_truth_time > latest_possible_time:
                break
            self._how_many_ground_truth_words += 1
        self._how_many_false_positives = 0
        self._how_many_correct_words = 0
        self._how_many_wrong_words = 0
        has_ground_truth_been_matched = []
        for found_word in found_words:
            found_label = found_word[0]
            found_time = found_word[1]
            # print(found_time)
            earliest_time = found_time - time_tolerance_ms
            latest_time = found_time + time_tolerance_ms
            has_matched_been_found = False
            for ground_truth in self._ground_truth_occurence:
                ground_truth_time = ground_truth[1]
                if ground_truth_time > latest_time or ground_truth_time > latest_possible_time:
                    break
                if ground_truth_time < earliest_time:
                    continue
                ground_truth_label = ground_truth[0]
                if ground_truth_label == found_label and \
                        has_ground_truth_been_matched.count(ground_truth_time) == 0:
                    self._how_many_correct_words += 1
                else:
                    self._how_many_wrong_words += 1
                has_ground_truth_been_matched.append(ground_truth_time)
                has_matched_been_found = True
                break
            if not has_matched_been_found:
                self._how_many_false_positives += 1
        self._how_many_ground_truth_matched = len(has_ground_truth_been_matched)

    # Write a human-readable description of the statistics to stdout
    def print_accuracy_stats(self):
        if self._how_many_ground_truth_words == 0:
            logging.info("No ground truth yet, " + str(self._how_many_false_positives) +
                         " false positives")
        else:
            any_match_percentage = self._how_many_ground_truth_matched / \
                                   self._how_many_ground_truth_words * 100
            correct_match_percentage = self._how_many_correct_words / \
                                       self._how_many_ground_truth_words * 100
            wrong_match_percentage = self._how_many_wrong_words / \
                                     self._how_many_ground_truth_words * 100
            false_positive_percentage = self._how_many_false_positives / \
                                        self._how_many_ground_truth_words * 100
            logging.info(str(round(any_match_percentage, 2)) + "% matched, " +
                         str(round(correct_match_percentage, 2)) + "% correctly, " +
                         str(round(wrong_match_percentage, 2)) + "% wrongly, " +
                         str(round(false_positive_percentage, 2)) + "% false positives")
