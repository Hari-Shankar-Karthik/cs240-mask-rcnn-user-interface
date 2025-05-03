import styles from "../styles/ImageUploader.module.css";

interface ImageUploaderProps {
  onUpload: (file: File) => void;
  image: string | null;
  loading: boolean;
  error: string | null;
  results: {
    metrics: {
      original_edge_alignment_score: number;
      original_region_homogeneity_score: number;
      custom_edge_alignment_score: number;
      custom_region_homogeneity_score: number;
      processing_time: number;
    };
  } | null;
  currentMaskIndex: number;
  totalInstances: number;
  processedInstances: number;
  onPrevMask: () => void;
  onNextMask: () => void;
}

function ImageUploader({
  onUpload,
  image,
  loading,
  error,
  results,
  currentMaskIndex,
  totalInstances,
  processedInstances,
  onPrevMask,
  onNextMask,
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
      {processedInstances < totalInstances && totalInstances > 0 && (
        <p className={styles.progress}>
          Processing {processedInstances} of {totalInstances} instances...
        </p>
      )}
      {results && results.metrics && (
        <div className={styles.metrics}>
          <h3>Performance Metrics</h3>
          <table className={styles.metricsTable}>
            <thead>
              <tr>
                <th>Metric</th>
                <th>Original (Mask R-CNN)</th>
                <th>Custom (A* Refined)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Edge Alignment Score</td>
                <td>
                  {results.metrics.original_edge_alignment_score.toFixed(4)}
                </td>
                <td
                  className={
                    results.metrics.custom_edge_alignment_score >
                    results.metrics.original_edge_alignment_score
                      ? styles.improved
                      : results.metrics.custom_edge_alignment_score <
                        results.metrics.original_edge_alignment_score
                      ? styles.regressed
                      : styles.neutral
                  }
                >
                  {results.metrics.custom_edge_alignment_score.toFixed(4)}
                </td>
              </tr>
              <tr>
                <td>Region Homogeneity Score</td>
                <td>
                  {results.metrics.original_region_homogeneity_score.toFixed(4)}
                </td>
                <td
                  className={
                    results.metrics.custom_region_homogeneity_score >
                    results.metrics.original_region_homogeneity_score
                      ? styles.improved
                      : results.metrics.custom_region_homogeneity_score <
                        results.metrics.original_region_homogeneity_score
                      ? styles.regressed
                      : styles.neutral
                  }
                >
                  {results.metrics.custom_region_homogeneity_score.toFixed(4)}
                </td>
              </tr>
              <tr>
                <td>Processing Time</td>
                <td colSpan={2} className={styles.neutral}>
                  {results.metrics.processing_time.toFixed(2)} seconds
                </td>
              </tr>
            </tbody>
          </table>
          <div className={styles.navigation}>
            <button
              onClick={onPrevMask}
              disabled={currentMaskIndex === 0}
              className={styles.navButton}
            >
              Prev
            </button>
            <span className={styles.maskCounter}>
              {totalInstances > 0
                ? `${currentMaskIndex + 1} of ${totalInstances}`
                : "No instances detected"}
            </span>
            <button
              onClick={onNextMask}
              disabled={currentMaskIndex >= processedInstances - 1}
              className={styles.navButton}
            >
              Next
            </button>
          </div>
          {currentMaskIndex >= totalInstances - 1 && totalInstances > 0 && (
            <p className={styles.lastMaskMessage}>
              This is the last instance mask.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

export default ImageUploader;
