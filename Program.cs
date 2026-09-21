using Microsoft.ML;
using Microsoft.ML.Data;
using InsuranceRegression;

var context = new MLContext(seed: 42);

string dataPath = Path.Combine(AppContext.BaseDirectory, "Data", "insurance.csv");

IDataView dataView = context.Data.LoadFromTextFile<InsuranceData>(
    dataPath,
    hasHeader: true,
    separatorChar: ',');

DataOperationsCatalog.TrainTestData split = context.Data.TrainTestSplit(dataView, testFraction: 0.2);
IDataView trainData = split.TrainSet;
IDataView testData = split.TestSet;


var dataPrepPipeline = context.Transforms.Categorical.OneHotEncoding("SexEncoded", "Sex")
    .Append(context.Transforms.Categorical.OneHotEncoding("SmokerEncoded", "Smoker"))
    .Append(context.Transforms.Categorical.OneHotEncoding("RegionEncoded", "Region"))
    .Append(context.Transforms.Concatenate("Features",
        "Age", "Bmi", "Children", "SexEncoded", "SmokerEncoded", "RegionEncoded"))
    .Append(context.Transforms.CopyColumns("Label", "Charges"))
    .Append(context.Transforms.NormalizeMinMax("Features"));

var trainers = new Dictionary<string, IEstimator<ITransformer>>
{
    ["Sdca (лінійна регресія)"] = context.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features"),
    ["LightGbm"] = context.Regression.Trainers.LightGbm(labelColumnName: "Label", featureColumnName: "Features"),
    ["FastTreeTweedie"] = context.Regression.Trainers.FastTreeTweedie(labelColumnName: "Label", featureColumnName: "Features"),
    ["FastForest"] = context.Regression.Trainers.FastForest(labelColumnName: "Label", featureColumnName: "Features"),
    ["Ols"] = context.Regression.Trainers.Ols(labelColumnName: "Label", featureColumnName: "Features"),
};

ITransformer? bestModel = null;
string bestName = string.Empty;
double bestR2 = double.MinValue;

Console.WriteLine("Порівняння регресійних моделей на варіанті 5 (прогноз медичного страхування)");
Console.WriteLine(new string('-', 75));

foreach (var (name, trainer) in trainers)
{
    var pipeline = dataPrepPipeline.Append(trainer);
    ITransformer model = pipeline.Fit(trainData);

    IDataView predictions = model.Transform(testData);
    RegressionMetrics metrics = context.Regression.Evaluate(predictions, labelColumnName: "Label");

    Console.WriteLine($"{name,-28} R^2 = {metrics.RSquared:F4}   MAE = {metrics.MeanAbsoluteError,10:F2}   RMSE = {metrics.RootMeanSquaredError,10:F2}");

    if (metrics.RSquared > bestR2)
    {
        bestR2 = metrics.RSquared;
        bestModel = model;
        bestName = name;
    }
}

Console.WriteLine(new string('-', 75));
Console.WriteLine($"Найкраща модель: {bestName} (R^2 = {bestR2:F4})");


if (bestModel is not null)
{
    var predictionEngine = context.Model.CreatePredictionEngine<InsuranceData, InsurancePrediction>(bestModel);

    var sample = new InsuranceData
    {
        Age = 35,
        Sex = "male",
        Bmi = 28.5f,
        Children = 2,
        Smoker = "yes",
        Region = "southeast"
    };

    InsurancePrediction result = predictionEngine.Predict(sample);
    Console.WriteLine();
    Console.WriteLine("Приклад прогнозу для нового клієнта:");
    Console.WriteLine($"  Вік=35, стать=male, ІМТ=28.5, дітей=2, курець=yes, регіон=southeast");
    Console.WriteLine($"  Прогнозовані витрати на страхування: {result.PredictedCharges:F2} $");

  
    string modelPath = Path.Combine(AppContext.BaseDirectory, "insurance_model.zip");
    context.Model.Save(bestModel, trainData.Schema, modelPath);
    Console.WriteLine($"\nМодель збережено у: {modelPath}");
}