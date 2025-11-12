using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;

namespace MlTrash
{
    public class WasteData
    {
        public string? ImagePath { get; set; }
        public string? Label { get; set; }
    }

    public class WastePrediction
    {
        [ColumnName("PredictedLabel")]
        public string? PredictedLabel { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            var mlContext = new MLContext(seed: 1);

            // =========================
            // Step 1: Load Dataset
            // =========================
            string datasetFolder = "WasteDataset"; // ganti sesuai folder dataset
            var images = Directory.GetDirectories(datasetFolder)
                .SelectMany(labelDir =>
                    Directory.GetFiles(labelDir, "*.*", SearchOption.AllDirectories)
                        .Where(f => f.EndsWith(".jpg") || f.EndsWith(".png"))
                        .Select(file => new WasteData
                        {
                            ImagePath = file,
                            Label = Path.GetFileName(labelDir)
                        })
                ).ToList();

            var data = mlContext.Data.LoadFromEnumerable(images);

            // =========================
            // Step 2: Split Data
            // =========================
            var trainTestData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // =========================
            // Step 3: Training Pipeline
            // =========================
            var pipeline = mlContext.MulticlassClassification.Trainers
                .ImageClassification(new ImageClassificationTrainer.Options()
                {
                    FeatureColumnName = "ImagePath",
                    LabelColumnName = "Label",
                    ValidationSet = testData,
                    Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                    Epoch = 10, // versi cepat
                    BatchSize = 10,
                    LearningRate = 0.01f,
                    MetricsCallback = (metrics) => Console.WriteLine(metrics),
                    ReuseTrainSetBottleneckCachedValues = true,
                    ReuseValidationSetBottleneckCachedValues = true
                })
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.WriteLine("Training model...");
            var model = pipeline.Fit(trainData);

            // =========================
            // Step 4: Evaluate Model
            // =========================
            var predictions = model.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label");

            Console.WriteLine("\n=== Evaluasi Model ===");
            Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:F2}");
            Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy:F2}");
            Console.WriteLine($"LogLoss: {metrics.LogLoss:F2}");

            // =========================
            // Step 5: Save Model
            // =========================
            string modelPath = "WasteClassificationModel.zip";
            mlContext.Model.Save(model, trainData.Schema, modelPath);
            Console.WriteLine($"\nModel tersimpan di {modelPath}");

            // =========================
            // Step 6: Batch Prediksi
            // =========================
            string testFolder = "TestImages"; // folder gambar baru
            var imageFiles = Directory.GetFiles(testFolder, "*.*", SearchOption.AllDirectories)
                                      .Where(f => f.EndsWith(".jpg") || f.EndsWith(".png"))
                                      .ToList();

            var predictor = mlContext.Model.CreatePredictionEngine<WasteData, WastePrediction>(model);

            var predictionsList = new List<string>();
            predictionsList.Add("ImageName,PredictedLabel");

            var classCounts = new Dictionary<string, int>();

            Console.WriteLine($"\nMenemukan {imageFiles.Count} gambar untuk prediksi:");
            foreach (var imagePath in imageFiles)
            {
                var input = new WasteData { ImagePath = imagePath };
                var prediction = predictor.Predict(input);
                Console.WriteLine($"Gambar: {Path.GetFileName(imagePath)} -> Prediksi: {prediction.PredictedLabel}");
                predictionsList.Add($"{Path.GetFileName(imagePath)},{prediction.PredictedLabel}");

                if (prediction.PredictedLabel != null)
                {
                    if (classCounts.ContainsKey(prediction.PredictedLabel))
                        classCounts[prediction.PredictedLabel]++;
                    else
                        classCounts[prediction.PredictedLabel] = 1;
                }
            }

            // Save CSV
            string csvPath = "Predictions.csv";
            File.WriteAllLines(csvPath, predictionsList);
            Console.WriteLine($"\nHasil prediksi disimpan di {csvPath}");

            // =========================
            // Step 7: Distribusi Kelas
            // =========================
            Console.WriteLine("\nDistribusi kelas prediksi:");
            foreach (var kvp in classCounts)
                Console.WriteLine($"{kvp.Key}: {kvp.Value}");
        }
    }
}
