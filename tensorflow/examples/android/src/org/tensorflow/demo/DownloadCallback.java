package org.tensorflow.demo;

import android.net.NetworkInfo;

/**
 * Created by ovidio on 4/10/17.
 */

public interface DownloadCallback<T> {
    interface Progress {
        int ERROR = -1;
        int CONNECT_SUCCESS = 0;
        int GET_INPUT_STREAM_SUCCESS = 1;
        int PROCESS_INPUT_STREAM_IN_PROGRESS = 2;
        int PROCESS_INPUT_STREAM_SUCCESS = 3;
    }

    /*
     * Indicates that the callback handler needs to update its appearance or information based on
     * the result of the task. Expected to be called from the main thread.
     */
    void updateFromDownload(T result);

    /*
     * Get the device's active network status in the form of a NetworkInfo object.
     */
    NetworkInfo getActiveNetworkInfo();

    /*
     * Indicate to callback handler any progress update.
     * @param progressCode must be one of the constants defined in the DownloadCallback.Progress.
     * @param percentComplete must be 0 - 100.
     */
    void onProgressUpdate(int progressCode, int percentComplete);

    /*
     * Indicates that the download operation has finished. This method is called even if the
     * download hasn't completed successfully.
     */
    void finishDownloading();
}
