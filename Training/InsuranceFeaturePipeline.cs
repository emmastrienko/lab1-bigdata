using Microsoft.ML;

namespace InsuranceRegression.Training;

// Побудова конвеєра підготовки ознак: one-hot кодування, об'єднання, нормалізація
public static class InsuranceFeaturePipeline
{
    public static IEstimator<ITransformer> Build(MLContext context)
    {
        return context.Transforms.Categorical.OneHotEncoding("SexEncoded", "Sex")
            .Append(context.Transforms.Categorical.OneHotEncoding("SmokerEncoded", "Smoker"))
            .Append(context.Transforms.Categorical.OneHotEncoding("RegionEncoded", "Region"))
            .Append(context.Transforms.Concatenate("Features",
                "Age", "Bmi", "Children", "SexEncoded", "SmokerEncoded", "RegionEncoded"))
            .Append(context.Transforms.CopyColumns("Label", "Charges"))
            .Append(context.Transforms.NormalizeMinMax("Features"));
    }
}
