package org.tensorflow.demo;

import android.content.Context;
import android.content.Intent;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentActivity;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;

public class MainActivity extends AppCompatActivity implements DownloadCallback {

    // Reference to NetworkFragment that executes network operations.
    private static String serverUrl = "http://192.168.1.102";
    private NetworkFragment mNetworkFragment;

    // Flag that is set when a download is in progress to prevent overlapping downloads
    // triggered by consecutive calls.
    private boolean mDownloading = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mNetworkFragment = NetworkFragment.getInstance(getSupportFragmentManager(), serverUrl);
    }

    /*
     * Change to the Classify activity when the button is pressed.
     */
    public void classify(View view) {
        Intent intent = new Intent(this, ClassifierActivity.class);
        startActivity(intent);
    }

    public void history(View view) {
        Intent intent = new Intent(this, ClassifyHistoryActivity.class);
        startActivity(intent);
    }

    /*
     * Add a menu to the view.
     */
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.camera_menu, menu);
        return true;
    }

    /*
     * Handle menu click events.
     */
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle item selection
        switch (item.getItemId()) {
            case R.id.update_model:
                updateModel();
                return true;
            case R.id.about:
                about();
                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }

    /*
     * Check if the server has a new update model update available and download it.
     */
    public void updateModel() {
        // Check if update exists.

        // Download updates if they exist.
        startDownload();
    }

    /*
     * Display the about view.
     */
    public void about() {
        // Change to about view
        Intent intent = new Intent(this, AboutActivity.class);
        startActivity(intent);
    }

    /*
     * Initiate the download.
     */
    private void startDownload() {
        if (!mDownloading && mNetworkFragment != null) {
            // Execute the async download.
            mNetworkFragment.startDownload();
            mDownloading = true;
        }
    }

    @Override
    public void updateFromDownload(Object result) {
        // Update UI based on result from download
        Log.d("Model update result", result.toString());
    }

    @Override
    public NetworkInfo getActiveNetworkInfo() {
        ConnectivityManager connectivityManager =
                (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        NetworkInfo networkInfo = connectivityManager.getActiveNetworkInfo();
        return networkInfo;
    }

    @Override
    public void onProgressUpdate(int progressCode, int percentComplete) {
        Log.d("Model update progress percentComplete", Integer.toString(percentComplete));
        switch (progressCode) {
            // You can add UI behavior for progress updates here.
            case Progress.ERROR:
                Log.d("Model update progress", "ERROR");
                break;
            case Progress.CONNECT_SUCCESS:
                Log.d("Model update progress", "CONNECT_SUCCESS");
                break;
            case Progress.GET_INPUT_STREAM_SUCCESS:
                Log.d("Model update progress", "GET_INPUT_STREAM_SUCCESS");
                break;
            case Progress.PROCESS_INPUT_STREAM_IN_PROGRESS:
                Log.d("Model update progress", "PROCESS_INPUT_STREAM_IN_PROGRESS");
                break;
            case Progress.PROCESS_INPUT_STREAM_SUCCESS:
                Log.d("Model update progress", "PROCESS_INPUT_STREAM_SUCCESS");
                break;
        }
    }

    @Override
    public void finishDownloading() {
        Log.d("Model update", "Finished downloading.");
        mDownloading = false;
        if (mNetworkFragment != null) {
            mNetworkFragment.cancelDownload();
        }
    }
}
