'use client';
import { useEffect, useState, useRef, ChangeEvent } from 'react';
import { Tensor, InferenceSession } from 'onnxruntime-web';

interface DetectionResult {
  label: string;
  confidence: number;
}

export default function YOLO() {
  const [session, setSession] = useState<{ net: InferenceSession } | null>(
    null
  );
  const [detectionResult, setDetectionResult] = useState<
    DetectionResult[] | null
  >(null);

  const ONNXModel = '/model/yolov8.onnx';
  const ModelShapes = [1, 3, 256, 256];

  async function LoadModel() {
    const yolov8 = await InferenceSession.create(ONNXModel, {
      executionProviders: ['wasm'],
    });
    const tensor = new Tensor(
      'float32',
      new Float32Array(ModelShapes.reduce((a, b) => a * b)),
      ModelShapes
    );
    await yolov8.run({ images: tensor });
    setSession({ net: yolov8 });
    console.log('Model Loaded!');
  }

  function handleImageUpload(event: ChangeEvent<HTMLInputElement>) {
    const { files } = event.target;
    if (files === null) {
      throw new Error('ちゃんと画像入れろよ');
    } else {
      const file = files[0];
      if (!file) return;
      const image = new Image();
      image.src = URL.createObjectURL(file);
      image.onload = async () => {};
    }
  }
}
