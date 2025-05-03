import { useState, useEffect } from "react";
import ImageUploader from "../components/ImageUploader";
import { API_URL } from "../constants";
import styles from "../styles/Home.module.css";

interface Results {
  original_mask: string;
  custom_mask: string;
  metrics: {
    original_edge_alignment_score: number;
    original_region_homogeneity_score: number;
    custom_edge_alignment_score: number;
    custom_region_homogeneity_score: number;
    processing_time: number;
  };
  total_instances: number;
}

function Home() {
  const [image, setImage] = useState<string | null>(null);
  const [imageId, setImageId] = useState<string | null>(null);
  const [currentMaskIndex, setCurrentMaskIndex] = useState<number>(0);
  const [cachedResults, setCachedResults] = useState<(Results | null)[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [totalInstances, setTotalInstances] = useState<number>(0);
  const [processedInstances, setProcessedInstances] = useState<number>(0);

  const handleImageUpload = async (file: File) => {
    if (!file) return;

    setImage(URL.createObjectURL(file));
    setLoading(true);
    setError(null);
    setCachedResults([]);
    setImageId(null);
    setCurrentMaskIndex(0);
    setTotalInstances(0);
    setProcessedInstances(0);

    const formData = new FormData();
    formData.append("image", file);
    formData.append("index", "0");

    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setImageId(data.image_id);
        setCachedResults([data.results]);
        setTotalInstances(data.results.total_instances);
        setProcessedInstances(1);
        setLoading(false);
      } else {
        setError(data.error || "Failed to upload image");
        setLoading(false);
      }
    } catch (err) {
      setError("Network error during upload");
      setLoading(false);
    }
  };

  const fetchMask = async (index: number) => {
    if (!imageId || cachedResults[index]) return; // Use cached result

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/results/${imageId}/${index}`);
      const data = await response.json();
      if (response.ok) {
        setCachedResults((prev) => {
          const newResults = [...prev];
          newResults[index] = data;
          return newResults;
        });
        setProcessedInstances((prev) => Math.max(prev, index + 1));
        setLoading(false);
      } else {
        setError(data.error || "Failed to fetch mask");
        setLoading(false);
      }
    } catch (err) {
      setError("Network error while fetching mask");
      setLoading(false);
    }
  };

  // Poll for new results every 5 seconds
  useEffect(() => {
    if (!imageId || processedInstances >= totalInstances) return;

    const pollResults = async () => {
      for (let index = processedInstances; index < totalInstances; index++) {
        if (cachedResults[index]) continue;
        try {
          const response = await fetch(
            `${API_URL}/results/${imageId}/${index}`
          );
          if (response.ok) {
            const data = await response.json();
            setCachedResults((prev) => {
              const newResults = [...prev];
              newResults[index] = data;
              return newResults;
            });
            setProcessedInstances((prev) => Math.max(prev, index + 1));
          }
        } catch (err) {
          // Silently ignore errors during polling
        }
      }
    };

    const interval = setInterval(pollResults, 5000);
    return () => clearInterval(interval);
  }, [imageId, processedInstances, totalInstances, cachedResults]);

  useEffect(() => {
    if (imageId && currentMaskIndex >= 0) {
      fetchMask(currentMaskIndex);
    }
  }, [imageId, currentMaskIndex]);

  const handlePrevMask = () => {
    setCurrentMaskIndex((prev) => Math.max(0, prev - 1));
  };

  const handleNextMask = () => {
    setCurrentMaskIndex((prev) => prev + 1);
  };

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
          results={cachedResults[currentMaskIndex]}
          currentMaskIndex={currentMaskIndex}
          totalInstances={totalInstances}
          processedInstances={processedInstances}
          onPrevMask={handlePrevMask}
          onNextMask={handleNextMask}
        />
        <div className={styles.maskContainer}>
          <div className={styles.maskSection}>
            <h2>Original Mask R-CNN Output</h2>
            {cachedResults[currentMaskIndex]?.original_mask ? (
              <img
                src={`data:image/png;base64,${cachedResults[currentMaskIndex].original_mask}`}
                alt={`Original Mask R-CNN ${currentMaskIndex + 1}`}
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
            {cachedResults[currentMaskIndex]?.custom_mask ? (
              <img
                src={`data:image/png;base64,${cachedResults[currentMaskIndex].custom_mask}`}
                alt={`Custom A* Refined Mask ${currentMaskIndex + 1}`}
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
