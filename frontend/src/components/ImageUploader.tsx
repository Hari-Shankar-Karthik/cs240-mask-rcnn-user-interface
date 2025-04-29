import styles from "../styles/ImageUploader.module.css";

interface ImageUploaderProps {
  onUpload: (file: File) => void;
  image: string | null;
  loading: boolean;
  error: string | null;
  results: {
    metrics: {
      iou_improvement: number;
      dice_coefficient: number;
      processing_time: number;
    };
  } | null;
}

function ImageUploader({
  onUpload,
  image,
  loading,
  error,
  results,
}: ImageUploaderProps) {
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onUpload(file);
  };

  return (
    <div className={styles.container}>
      <h2>Upload Image</h2>
      <input
        type="file"
        accept="image/*"
        onChange={handleImageUpload}
        className={styles.fileInput}
      />
      {image && (
        <div className={styles.imageContainer}>
          <img
            src={image}
            alt="Uploaded Image"
            className={styles.uploadedImage}
          />
        </div>
      )}
      {loading && (
        <p className={styles.loading}>Processing image, please wait...</p>
      )}
      {error && <p className={styles.error}>{error}</p>}
      {results && (
        <div className={styles.metrics}>
          <h3>Performance Metrics</h3>
          <div className={styles.metricsContent}>
            <p>
              <span>IoU Improvement:</span>{" "}
              {results.metrics.iou_improvement.toFixed(4)}
            </p>
            <p>
              <span>Dice Coefficient:</span>{" "}
              {results.metrics.dice_coefficient.toFixed(4)}
            </p>
            <p>
              <span>Processing Time:</span>{" "}
              {results.metrics.processing_time.toFixed(2)} seconds
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

export default ImageUploader;
