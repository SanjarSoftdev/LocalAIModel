using Microsoft.SemanticKernel;
using LocalAIModel.Utils;
using feiyun0112.SemanticKernel.Connectors.OnnxRuntimeGenAI;
using Microsoft.SemanticKernel.Connectors.OpenAI;

//ConsoleHelper.ShowHeader();
string modelPath = Statics.ModelPath;
    //= ConsoleHelper.GetFolderPath(Statics.ModelInputPrompt);

//ConsoleHelper.ShowHeader();
ConsoleHelper.WriteToConsole(Statics.ModelLoadingMessage);

var kernelBuilder = Kernel.CreateBuilder();

Kernel kernel = kernelBuilder
           .AddOnnxRuntimeGenAIChatCompletion(modelPath: modelPath)
           .Build();
kernel.ImportPluginFromType<PersonAges>();
var systemPrompt = "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users.";


var key2 = Console.ReadKey();
while (true)
{
    // Get user question
    ConsoleHelper.WriteToConsole(Environment.NewLine);
    var userQ = ConsoleHelper.Getprompt(Statics.UserPrompt);
    //ConsoleHelper.ShowHeader();
    ConsoleHelper.WriteToConsole(Statics.Execute_Task);
    ConsoleHelper.WriteToConsole(Environment.NewLine);

    if (string.IsNullOrEmpty(userQ))
    {
        break;
    }

    // show phi3 response
    ConsoleHelper.WriteToConsole(Statics.PhiChat);

    OpenAIPromptExecutionSettings executionSettings = new()
    {
        ToolCallBehavior = ToolCallBehavior.AutoInvokeKernelFunctions
    };

    var fullPrompt = $"<|system|>{systemPrompt}<|end|><|user|>{userQ}<|end|><|assistant|>";
    await foreach (string text in kernel.InvokePromptStreamingAsync<string>(fullPrompt,
                   new KernelArguments(executionSettings)))
    {
        if (text.Contains("</s>"))
        {
            break;
        }
        ConsoleHelper.WriteToConsole(text);
    }
    ConsoleHelper.WriteToConsole(Environment.NewLine);
    ConsoleHelper.WriteToConsole(Statics.RestartPrompt);
    ConsoleHelper.WriteToConsole(Environment.NewLine);
    var key = Console.ReadKey();
    if (key.Key == ConsoleKey.Escape)
    {
        break;
    }
}

class PersonAges
{
    [KernelFunction]
    public int GetPersonAge(string name)
    {
        return name switch
        {
            "Sanjar" => 22,
            "Dilshod" => 21,
            "Azimjon" => 22,
            _ => 30
        };
    }
}