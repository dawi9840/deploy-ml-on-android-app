package com.example.tflite_deploy_on_android

import android.annotation.SuppressLint
import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import com.example.tflite_deploy_on_android.databinding.ActivityIrisBinding
import com.example.tflite_deploy_on_android.ml.QuantiziedIris
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.ByteBuffer

class IrisActivity : AppCompatActivity() {

    private lateinit var binding: ActivityIrisBinding
    private lateinit var btnHome: Button
    private lateinit var btnClassify: Button
    private lateinit var editTxt1: EditText
    private lateinit var editTxt2: EditText
    private lateinit var editTxt3: EditText
    private lateinit var editTxt4: EditText
    private  lateinit var txtView: TextView

    @SuppressLint("SetTextI18n")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityIrisBinding.inflate(layoutInflater)
        setContentView(binding.root)

        btnHome = binding.button3
        btnClassify = binding.button

        editTxt1 = binding.editTextNumberDecimal1
        editTxt2 = binding.editTextNumberDecimal2
        editTxt3 = binding.editTextNumberDecimal3
        editTxt4 = binding.editTextNumberDecimal4

        txtView = binding.textView

        btnHome.setOnClickListener(View.OnClickListener{
            val intent = Intent(this, MainActivity::class.java)
            startActivity(intent)
        })

        btnClassify.setOnClickListener(View.OnClickListener{
            val v1 : Float = editTxt1.text.toString().toFloat()
            val v2 : Float = editTxt2.text.toString().toFloat()
            val v3 : Float = editTxt3.text.toString().toFloat()
            val v4 : Float = editTxt4.text.toString().toFloat()

            val byteBuffer : ByteBuffer = ByteBuffer.allocate(4*4)
            byteBuffer.putFloat(v1)
            byteBuffer.putFloat(v2)
            byteBuffer.putFloat(v3)
            byteBuffer.putFloat(v4)

            val model = QuantiziedIris.newInstance(this)

            // Creates inputs for reference.
            // The input size was 4 column and best size, but in this case it's only one by
            // size so we have to create area of one by four.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 4), DataType.FLOAT32)
            inputFeature0.loadBuffer(byteBuffer)

            // Runs model inference and gets result.
            val outputs = model.process(inputFeature0)

            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

            txtView.text = "Iris-setosa: " + outputFeature0[0].toString() + "\n" +
                    "Iris-versicolor: "+ outputFeature0[1].toString() + "\n" +
                    "Iris-virginica: " + outputFeature0[2].toString()

            // Releases model resources if no longer used.
            model.close()
        })

    }
}