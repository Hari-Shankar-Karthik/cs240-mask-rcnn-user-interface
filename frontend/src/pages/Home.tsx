import { useState, useEffect } from "react";
import ImageUploader from "../components/ImageUploader";
import { API_URL } from "../constants";
import styles from "../styles/Home.module.css";

interface Results {
  original_mask: string;
  custom_mask: string;
  metrics: {
    iou_improvement: number;
    dice_coefficient: number;
    processing_time: number;
  };
}

function Home() {
  const [image, setImage] = useState<string | null>(null);
  const [imageId, setImageId] = useState<string | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleImageUpload = async (file: File) => {
    if (!file) return;

    setImage(URL.createObjectURL(file));
    setLoading(true);
    setError(null);
    setResults(null);
    setImageId(null);

    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setImageId(data.image_id);
      } else {
        setError(data.error || "Failed to upload image");
        setLoading(false);
      }
    } catch (err) {
      setError("Network error during upload");
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!imageId) return;

    const pollResults = async () => {
      try {
        const response = await fetch(`${API_URL}/results/${imageId}`);
        const data = await response.json();
        if (response.ok) {
          setResults(data);
          setLoading(false);
        } else if (response.status === 202) {
          setTimeout(pollResults, 2000); // Poll every 2 seconds
        } else {
          setError(data.error || "Failed to fetch results");
          setLoading(false);
        }
      } catch (err) {
        setError("Network error while fetching results");
        setLoading(false);
      }
    };

    pollResults();
  }, [imageId]);

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <h1>Mask Refinement Interface</h1>
      </header>
      <main className={styles.main}>
        <ImageUploader
          onUpload={handleImageUpload}
          image={image}
          loading={loading}
          error={error}
          results={results}
        />
        <div className={styles.maskContainer}>
          <div className={styles.maskSection}>
            <h2>Original Mask R-CNN Output</h2>
            {results && results.original_mask ? (
              <img
                src={`data:image/png;base64,${results.original_mask}`}
                alt="Original Mask R-CNN"
                className={styles.maskImage}
              />
            ) : (
              <div className={styles.placeholder}>
                <p>No mask available</p>
              </div>
            )}
          </div>
          <div className={styles.maskSection}>
            <h2>Custom A* Refined Output</h2>
            {results && results.custom_mask ? (
              <img
                src={`data:image/png;base64,${results.custom_mask}`}
                alt="Custom A* Refined Mask"
                className={styles.maskImage}
              />
            ) : (
              <div className={styles.placeholder}>
                <p>No mask available</p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default Home;
