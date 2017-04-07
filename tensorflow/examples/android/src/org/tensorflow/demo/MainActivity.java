package org.tensorflow.demo;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    /*
     * Change to the Classify activity when the button is pressed.
     */
    public void classify(View view) {
        Intent intent = new Intent(this, ClassifierActivity.class);
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
    }

    /*
     * Display the about view.
     */
    public void about() {
        // Change to about view
        Intent intent = new Intent(this, AboutActivity.class);
        startActivity(intent);
    }
}
