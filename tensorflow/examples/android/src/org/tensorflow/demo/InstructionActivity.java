package org.tensorflow.demo;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;

/**
 * Created by dennis on 4/26/17.
 */

public class InstructionActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_instructions);
        TextView textView = (TextView) findViewById(R.id.classify_instruction_body);

        String classify_instr = "   - Click Classify in menu.\n" +
                                "   - Wait for the camera screen to load.\n" +
                                "   - Top white board is where the results will show.\n"+
                                "   - You can choose between 4 models using the options on bottom right.\n" +
                                "   - You can also focus in on the leaf by tapping on the screen to where you want to focus.\n"+
                                "   - When you ready, click the image icon in the bottom middle.\n" +
                                "   - It will take a few second before the output is displayed in the white board at top. \n" +
                                "   - You may continue to take more images once the output is displayed.";
        textView.setText(classify_instr); //set text for text view

        textView = (TextView) findViewById(R.id.capture_instruction_body);

        String capture_instr = "    - Click Capture option from menu.\n" +
                               "    - When the screen is loaded click capture in bottom center to take the picture\n" +
                                "   - In the next screen give the image a name.\n"+
                                "   - Click upload photo to upload the image.";

        textView.setText(capture_instr);

        String Update_Model = "    - Click the 3 dotted menu at top right and click update model from the options.";

        textView = (TextView) findViewById(R.id.update_model_instruction_body);
        textView.setText(Update_Model);
    }
}
