package com.example.tflite_deploy_on_android

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.Button
import com.example.tflite_deploy_on_android.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var btnIris: Button
    private lateinit var btnImgClassifier: Button
    private lateinit var btnDigitClassifier: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        btnIris = binding.button2
        btnImgClassifier = binding.button5
        btnDigitClassifier = binding.button7

        btnIris.setOnClickListener(View.OnClickListener{
            val intent = Intent(this, IrisActivity::class.java)
            startActivity(intent)
        })
        btnImgClassifier.setOnClickListener(View.OnClickListener{
            val intent = Intent(this, ImgClassifierActivity::class.java)
            startActivity(intent)
        })
        btnDigitClassifier.setOnClickListener(View.OnClickListener{
            val intent = Intent(this, DigitClassifierActivity::class.java)
            startActivity(intent)
        })
    }
}