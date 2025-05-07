#include<iostream>
#include<torch/script.h>
#include<vector>
#include<string>
#include<sstream>
#include<algorithm>
#include<cmath>
#include<fstream>
#include<torch/torch.h>


torch::Tensor tokenize_text2(const std::string& encodings_file) {
    std::ifstream inputFile(encodings_file);

    if (!inputFile.is_open()) {
        std::cerr << "Unable to open file" << std::endl;
        return torch::tensor(-1);
    }

    // Read the whole file into a string
    std::stringstream buffer;
    buffer << inputFile.rdbuf();

    // Close the file
    inputFile.close();


    std::string fileContent = buffer.str();
    // std::cout<<fileContent<<"\n";
    size_t inputIdsPos = fileContent.find("\'input_ids\':"); // Find the start of the input_ids field

    if (inputIdsPos == std::string::npos) {
        std::cerr << "input_ids not found!" << std::endl;
        return torch::tensor(-1);
    }

    size_t start = fileContent.find("[[", inputIdsPos) + 2;
    size_t end = fileContent.find("]]", start);

    std::string inputIdsStr = fileContent.substr(start, end - start);


    std::vector<int> flattened_input_ids;
    size_t pos = 0;
    while ((pos = inputIdsStr.find(",")) != std::string::npos) {
        std::string token = inputIdsStr.substr(0, pos);
        flattened_input_ids.push_back(std::stoi(token));
        inputIdsStr.erase(0, pos + 1);
    }
    flattened_input_ids.push_back(std::stoi(inputIdsStr)); // Add the last element
    // Step 5: Convert to a tensor
    auto options = torch::TensorOptions().dtype(torch::kInt64);
    auto tensor = torch::tensor(flattened_input_ids, options).unsqueeze(0);;

    // Print the tensor to verify
    // std::cout << "Tensor: " << tensor << std::endl;

    return tensor;


    

}

double compute_perplexity(torch::Tensor input_ids,torch::jit::script::Module model){
    
    int64_t max_length=1024;
    int64_t stride=512;
    int64_t seq_len=input_ids.size(1);

    double nll_sum=0;
    int64_t n_tokens=0;
    int64_t prev_end_loc=0;

    for(int64_t begin_loc=0;begin_loc<seq_len;begin_loc+=stride){
        int64_t end_loc=std::min(begin_loc+max_length,seq_len);
        int64_t trg_len=end_loc-prev_end_loc;

        torch::Tensor trunc_input_ids=input_ids.slice(1,begin_loc,end_loc);
        torch::Tensor target_ids=trunc_input_ids.clone();

        target_ids.index_put_({torch::indexing::Slice(),torch::indexing::Slice(0,target_ids.size(1)-trg_len)},-100);


        // std::cout<<"input_ids: "<<trunc_input_ids<<std::endl;
        // std::cout<<std::endl;
        // std::cout<<"target_ids: "<<target_ids<<std::endl;


        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(trunc_input_ids);
        inputs.push_back(target_ids);
        auto output = model.forward(inputs).toTuple();
        auto loss = output->elements()[0].toTensor();
        // std::cout<<loss<<std::endl;
        double neg_log_likelihood = loss.item<double>();
        int64_t num_valid_tokens=(target_ids!=-100).sum().item<int64_t>();
        int64_t batch_size = target_ids.size(0);
        int64_t num_loss_tokens = num_valid_tokens - batch_size;

        nll_sum += neg_log_likelihood * num_loss_tokens;
        n_tokens += num_loss_tokens;

        prev_end_loc = end_loc;
        if (end_loc == seq_len) {
            break;
        }

    } 

    double avg_nll = nll_sum / n_tokens;
    double ppl = std::exp(avg_nll);

    std::cout << "Perplexity: " << ppl << std::endl;
    std::cout << "Number of tokens: " << n_tokens << std::endl;

    return ppl;
}

int main(){
    std::string model_path="/mnt/combined/home/parveen/gouri/traced_llama_4064_3.pt";
    std::string encodings_path="/mnt/combined/home/parveen/final-models/text-generation/cpp/encodings.txt";
    torch::jit::script::Module model;

    try{
        model=torch::jit::load(model_path);
    }

    catch(const c10::Error& e){
        std::cerr << "error loading the model\n";
        return -1;
    }
    
    std::cout<<"ok\n";

    torch::Tensor input_ids=tokenize_text2(encodings_path); 
    // std::vector<torch::jit::IValue> inputs;
    // inputs.push_back(input_ids);  // First input: input_ids
    // inputs.push_back(input_ids);

    // at::IValue outputs=model.forward({input_ids});
    
    // if (outputs.isTuple()) {
    //     auto output_tuple = outputs.toTuple();  // Get the tuple from IValue

    //     // Extract individual tensors from the tuple
    //     auto tensor_0 = output_tuple->elements()[0];
    //     auto tensor_1 = output_tuple->elements()[1];
    //     auto tensor_2 = output_tuple->elements()[2];

    //     // Print the tensors
    //     std::cout << "Output Tensor 0: " << tensor_0.toTensor().item() << std::endl;
    // } else {
    //     std::cerr << "Expected a tuple output, but got something else!" << std::endl;
    // }

    double ppl=compute_perplexity(input_ids,model);
    
    return 0;

    // std::cout<<"Hello World";
}