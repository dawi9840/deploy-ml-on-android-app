package com.example.tflite_deploy_on_android

import android.annotation.SuppressLint
import android.content.Intent
import android.graphics.Color
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import android.widget.TextView
import com.divyanshu.draw.widget.DrawView
import com.example.tflite_deploy_on_android.databinding.ActivityDigitClassifierBinding



class DigitClassifierActivity : AppCompatActivity() {
    private  lateinit var binding: ActivityDigitClassifierBinding
    private lateinit var predictTxt:TextView
    private lateinit var drawView: DrawView
    private lateinit var btnHome: Button
    private lateinit var btnClear: Button
    private var digitClassifier = DigitClassifier(this)

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityDigitClassifierBinding.inflate(layoutInflater)
        setContentView(binding.root)

        btnHome = binding.button6
        btnClear = binding.clearButton
        predictTxt = binding.predictedText

        // Setup view instances.
        drawView = binding.drawView
        drawView.setStrokeWidth(70.0f)
        drawView.setColor(Color.WHITE)
        drawView.setBackgroundColor(Color.BLACK)

        btnHome.setOnClickListener(View.OnClickListener{
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        })

        // Setup clear drawing button.
        btnClear.setOnClickListener {
            drawView.clearCanvas()
            predictTxt.text = getString(R.string.prediction_text_placeholder)
        }

        // Setup classification trigger so that it classify after every stroke drew.
        drawView.setOnTouchListener { _, event ->
            // As we have interrupted DrawView's touch event,
            // we first need to pass touch events through to the instance for the drawing to show up.
            drawView.onTouchEvent(event)

            // Then if user finished a touch event, run classification
            if (event.action == MotionEvent.ACTION_UP) {classifyDrawing()}

            true
        }

        // Setup digit classifier.
        digitClassifier
            .initialize()
            .addOnFailureListener { e -> Log.e(TAG, "Error to setting up digit classifier.", e) }
    }

    override fun onDestroy() {
        // Sync DigitClassifier instance lifecycle with MainActivity lifecycle,
        // and free up resources (e.g. TF Lite instance) once the activity is destroyed.
        digitClassifier.close()
        super.onDestroy()
    }

    @SuppressLint("StringFormatInvalid")
    private fun classifyDrawing() {
        val bitmap = drawView.getBitmap()

        if ((bitmap != null) && (digitClassifier.isInitialized)) {
            digitClassifier
                .classifyAsync(bitmap)
                .addOnSuccessListener { resultText -> predictTxt.text = resultText }
                .addOnFailureListener { e ->
                    predictTxt.text = getString(
                        R.string.classification_error_message,
                        e.localizedMessage
                    )
                    Log.e(TAG, "Error classifying drawing.", e)
                }
        }
    }

    companion object {
        private const val TAG = "DigitClassifierActivity"
    }
}