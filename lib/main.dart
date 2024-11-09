// pubspec.yaml
// dependencies:
//   tflite_flutter: ^0.9.0
//   image_picker: ^0.8.4+4

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: TFLiteHome(),
    );
  }
}

class TFLiteHome extends StatefulWidget {
  @override
  _TFLiteHomeState createState() => _TFLiteHomeState();
}

class _TFLiteHomeState extends State<TFLiteHome> {
  late Interpreter _interpreter;
  File? _image;
  List<String>? _results;
  List<String> _labels = [];
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    loadModel();
    loadLabels();
  }

  // โหลด labels จากไฟล์
  Future<void> loadLabels() async {
    try {
      String labelsData = await DefaultAssetBundle.of(context)
          .loadString('assets/carmlcoco.txt');
      setState(() {
        _labels = labelsData.split('\n');
      });
      print('Labels loaded successfully');
    } catch (e) {
      print('Error loading labels: $e');
    }
  }

  // โหลด TFLite model
  Future<void> loadModel() async {
    try {
      print(await rootBundle.load('assets/carml.tflite'));
      _interpreter = await Interpreter.fromAsset('assets/carml.tflite');
      print('Model loaded successfully');
    } catch (e) {
      print('Error loading model: $e');
    }
  }

  // ฟังก์ชันสำหรับการ inference
  Future<void> runInference() async {
    if (_image == null) return;

    setState(() {
      _isLoading = true;
    });

    try {
      // แปลงรูปภาพเป็น input tensor
      // Note: ต้องปรับ input shape และ preprocessing ตาม model ที่ใช้
      var inputShape = _interpreter.getInputTensor(0).shape;
      var outputShape = _interpreter.getOutputTensor(0).shape;

      // สร้าง input และ output buffers
      var inputBuffer = List.filled(inputShape[0] * inputShape[1] * inputShape[2], 0.0);
      var outputBuffer = List.filled(outputShape[0], 0.0);

      // รัน inference
      _interpreter.run(inputBuffer, outputBuffer);

      // แปลงผลลัพธ์เป็น labels
      List<MapEntry<double, String>> labeledResults = [];
      for (var i = 0; i < outputBuffer.length; i++) {
        if (i < _labels.length) {
          labeledResults.add(MapEntry(outputBuffer[i], _labels[i]));
        }
      }

      // เรียงลำดับตามความน่าจะเป็น
      labeledResults.sort((a, b) => b.key.compareTo(a.key));

      // แสดงผลลัพธ์ top 3
      setState(() {
        _results = labeledResults.take(3).map((entry) {
          var percentage = (entry.key * 100).toStringAsFixed(1);
          return '${entry.value}: $percentage%';
        }).toList();
      });

    } catch (e) {
      print('Error running inference: $e');
      setState(() {
        _results = ['Error processing image'];
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  // เลือกรูปภาพจากแกลเลอรี่
  Future<void> pickImage() async {
    final ImagePicker picker = ImagePicker();
    final XFile? pickedFile = await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _results = null; // รีเซ็ตผลลัพธ์เก่า
      });
      runInference();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('TFLite Image Classifier'),
      ),
      body: SingleChildScrollView(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (_image != null) ...[
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Image.file(
                    _image!,
                    height: 200,
                  ),
                ),
                SizedBox(height: 20),
              ],
              ElevatedButton.icon(
                onPressed: pickImage,
                icon: Icon(Icons.image),
                label: Text('Pick Image'),
              ),
              SizedBox(height: 20),
              if (_isLoading)
                CircularProgressIndicator()
              else if (_results != null)
                Card(
                  margin: EdgeInsets.all(8),
                  child: Padding(
                    padding: EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Results:',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        SizedBox(height: 8),
                        ..._results!.map((result) => Padding(
                          padding: EdgeInsets.symmetric(vertical: 4),
                          child: Text(
                            result,
                            style: TextStyle(fontSize: 16),
                          ),
                        )).toList(),
                      ],
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _interpreter.close();
    super.dispose();
  }
}