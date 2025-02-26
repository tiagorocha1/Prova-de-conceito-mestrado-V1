import React, { useEffect, useRef, useState } from 'react';
import { FaceDetection, Results } from '@mediapipe/face_detection';
import { Camera } from '@mediapipe/camera_utils';

interface MediaPipeDetection {
  boundingBox: {
    xCenter: number;
    yCenter: number;
    width: number;
    height: number;
    rotation?: number; // Pode estar indefinido
  };
  // Opcionalmente inclui "confidence" para cada landmark.
  landmarks: Array<{ x: number; y: number; z: number } & Partial<{ confidence: number }>>;
}

function isGoodQuality(det: MediaPipeDetection): boolean {
  if (det.boundingBox.rotation !== undefined && Math.abs(det.boundingBox.rotation) > 15) {
    return false;
  }
  if (det.landmarks.length < 6) {
    return false;
  }
  const eye1 = det.landmarks[0];
  const eye2 = det.landmarks[1];
  const dx = eye1.x - eye2.x;
  const dy = eye1.y - eye2.y;
  const interocularDistance = Math.sqrt(dx * dx + dy * dy);
  if (interocularDistance < det.boundingBox.width * 0.3) {
    return false;
  }
  for (const lm of det.landmarks) {
    if (lm.confidence !== undefined && lm.confidence < 0.5) {
      return false;
    }
  }
  return true;
}

function FaceDetectionComponent() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [detector, setDetector] = useState<FaceDetection | null>(null);
  const cameraRef = useRef<Camera | null>(null);
  const lastRequestTimeRef = useRef<number>(0);
  const throttleInterval = 2000; // 500 ms
  const [isDetecting, setIsDetecting] = useState<boolean>(false);

  useEffect(() => {
    const faceDetection = new FaceDetection({
      locateFile: (file: string) =>
        `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`,
    });
    faceDetection.setOptions({
      model: 'short', // ou full - para detecção de faces mais distantes (modelo de alcance total)
      minDetectionConfidence: 0.5, // Valor entre 0 e 1 que determina o limiar mínimo de confiança para considerar uma detecção válida.
    });
    faceDetection.onResults(onResults);
    setDetector(faceDetection);
  }, []);

  useEffect(() => {
    if (detector && videoRef.current) {
      // Cria a instância da câmera, mas não a inicia automaticamente
      cameraRef.current = new Camera(videoRef.current, {
        onFrame: async () => {
          await detector.send({ image: videoRef.current! });
        },
        width: 640,
        height: 480,
      });
    }
  }, [detector]);

  const onResults = async (results: Results) => {
    if (!canvasRef.current || !videoRef.current) return;

    const now = Date.now();
    if (now - lastRequestTimeRef.current < throttleInterval) return;
    lastRequestTimeRef.current = now;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Atualiza o canvas com o frame atual
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.save();
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    if (!results.detections || results.detections.length === 0) return;

    // Filtra detecções de boa qualidade
    const detections = results.detections as MediaPipeDetection[];
    const validDetections = detections.filter(det => isGoodQuality(det));
    //const validDetections = detections;
    if (validDetections.length === 0) return;

    // Cria um array com as faces recortadas
    const facesPayload = validDetections.map((det) => {
      const { xCenter, yCenter, width, height } = det.boundingBox;
      const xMin = xCenter - width / 2;
      const xMax = xCenter + width / 2;
      const yMin = yCenter - height / 2;
      const yMax = yCenter + height / 2;

      const rectX = xMin * canvas.width;
      const rectY = yMin * canvas.height;
      const rectWidth = (xMax - xMin) * canvas.width;
      const rectHeight = (yMax - yMin) * canvas.height;

      // (Opcional) Desenha retângulo ao redor da face
      ctx.beginPath();
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'blue';
      ctx.rect(rectX, rectY, rectWidth, rectHeight);
      ctx.stroke();

      // Cria canvas temporário para capturar a face
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = rectWidth;
      tempCanvas.height = rectHeight;
      const tempCtx = tempCanvas.getContext('2d');
      if (!tempCtx) return null;
      tempCtx.drawImage(
        canvas,
        rectX,
        rectY,
        rectWidth,
        rectHeight,
        0,
        0,
        rectWidth,
        rectHeight
      );
      const base64Image = tempCanvas.toDataURL('image/png');

      return { image: base64Image,  timestamp: now };
    }).filter((face): face is {image: string; timestamp: number } => face !== null);

    // Envia as faces de forma assíncrona (o retorno não é usado)
    fetch('http://localhost:8000/recognize-batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ images: facesPayload }),
    }).catch(err => {
      // O erro pode ser ignorado ou logado
      console.error('Erro ao enviar faces:', err);
    });
  };

  // Função para alternar a detecção (iniciar/parar)
  const toggleDetection = () => {
    if (isDetecting) {
      // Pausa a detecção
      cameraRef.current?.stop();
      setIsDetecting(false);
    } else {
      // Inicia a detecção
      cameraRef.current?.start();
      setIsDetecting(true);
    }
  };

  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{ margin: '0 auto', maxWidth: '640px' }}>
        <div style={{ position: 'relative', display: 'inline-block' }}>
          <video ref={videoRef} style={{ display: 'none' }} />
          <canvas ref={canvasRef} style={{ width: '100%' }} />
        </div>
      </div>
      <div style={{ marginTop: '20px' }}>
        <button
          onClick={toggleDetection}
          style={{
            padding: '10px 20px',
            backgroundColor: isDetecting ? '#d93025' : '#4285F4',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            fontSize: '16px',
          }}
        >
          {isDetecting ? 'Parar Detecção' : 'Iniciar Detecção'}
        </button>
      </div>
    </div>
  );
}

export default FaceDetectionComponent;
