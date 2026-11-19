using InsuranceRegression.Data;
using InsuranceRegression.Evaluation;
using InsuranceRegression.Models;
using InsuranceRegression.Reporting;
using InsuranceRegression.Training;
using Microsoft.ML;

var context = new MLContext(seed: 42);

string dataPath = Path.Combine(AppContext.BaseDirectory, "Data", "insurance.csv");

// 1. Завантаження даних та розбиття на навчальну/тестову вибірки
var loader = new InsuranceDataLoader(context);
DataOperationsCatalog.TrainTestData split = loader.LoadAndSplit(dataPath);

// 2. Конвеєр підготовки ознак і набір тренерів
var prepPipeline = InsuranceFeaturePipeline.Build(context);
var trainers = RegressionTrainerCatalog.GetTrainers(context);

ConsoleReporter.PrintComparisonHeader();

// 3. Навчання та оцінювання кожної моделі
var evaluator = new RegressionEvaluator(context);
var results = new List<TrainerResult>();
foreach (var (name, trainer) in trainers)
{
    var result = evaluator.TrainAndEvaluate(name, prepPipeline.Append(trainer), split.TrainSet, split.TestSet);
    results.Add(result);
    ConsoleReporter.PrintComparisonRow(result);
}

// 4. Вибір найкращої моделі за R^2
var best = evaluator.SelectBest(results);
ConsoleReporter.PrintBest(best);

// 5. Приклад прогнозу для нового клієнта
if (best is not null)
{
    var predictionEngine = context.Model.CreatePredictionEngine<InsuranceData, InsurancePrediction>(best.Model);

    var sample = new InsuranceData
    {
        Age = 35,
        Sex = "male",
        Bmi = 28.5f,
        Children = 2,
        Smoker = "yes",
        Region = "southeast"
    };

    ConsoleReporter.PrintSamplePrediction(predictionEngine.Predict(sample));

    // 6. Збереження найкращої моделі
    string modelPath = Path.Combine(AppContext.BaseDirectory, "insurance_model.zip");
    context.Model.Save(best.Model, split.TrainSet.Schema, modelPath);
    Console.WriteLine($"\nМодель збережено у: {modelPath}");
}
