using Microsoft.ML;

namespace InsuranceRegression.Training;

// Каталог тренерів для порівняння
public static class RegressionTrainerCatalog
{
    public static Dictionary<string, IEstimator<ITransformer>> GetTrainers(MLContext context)
    {
        return new Dictionary<string, IEstimator<ITransformer>>
        {
            ["Sdca (лінійна регресія)"] = context.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features"),
            ["LightGbm"] = context.Regression.Trainers.LightGbm(labelColumnName: "Label", featureColumnName: "Features"),
            ["FastTreeTweedie"] = context.Regression.Trainers.FastTreeTweedie(labelColumnName: "Label", featureColumnName: "Features"),
            ["FastForest"] = context.Regression.Trainers.FastForest(labelColumnName: "Label", featureColumnName: "Features"),
            ["Ols"] = context.Regression.Trainers.Ols(labelColumnName: "Label", featureColumnName: "Features"),
        };
    }
}
