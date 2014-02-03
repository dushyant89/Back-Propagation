import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;


/**
 * @author Dushyant Sabharwal
 *
 */
public class Classifier_XOR {

	/**
	 * @param args
	 * @throws IOException 
	 * @throws NumberFormatException 
	 */
	public static void main(String[] args) throws NumberFormatException, IOException 
	{
		double e=2.71828182846;

		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		System.out.println("Enter the no. of inputs");
		int input=Integer.parseInt(br.readLine());
		int cols=input+1;
		int rows=(int)Math.pow(2,input);
		int [][]xor=new int[rows][cols];
		int number;
		//generating truth table for user provided input
		for(int i=0;i<rows;i++)
		{
			input=cols-1;
			number=i;
			while(number!=0)
			{
				xor[i][--input]=number%2;
				number=number/2;
			}
			while(input >0)
			{
				xor[i][--input]=0;
			}
		}
		//generating the xor table
		for(int i=0;i<rows;i++)
		{
			xor[i][cols-1]=0;
			for(int j=0;j<cols-1;j++)
			{
				xor[i][cols-1]^=xor[i][j];
			}
		}
		System.out.println("Enter the no of hidden layers");
		int hidden_layer=Integer.parseInt(br.readLine());


		System.out.println("Enter the no. of neurons per hidden layer");
		int no_neurons=Integer.parseInt(br.readLine());       // no of neurons per hidden layer

		System.out.println("Enter the learning rate");
		
		double scaling_factor=Double.parseDouble(br.readLine());
		
		System.out.println("Enter the momentum rate:Note for a high learning rate do not keep high momentum");
		double momentum=Double.parseDouble(br.readLine());

		int total_links=(cols)*no_neurons*hidden_layer+no_neurons;

		hidden_layer++; // for the output neuron at the last layer

		//creating links for the network
		Link[] links=new Link[total_links];

		for(int i=0;i<total_links;i++)
		{
			links[i]=new Link();
			links[i].weight=Math.random();
		}

		//creating layers
		Layer []layers=new Layer[hidden_layer];
		
		int link_tracker=0;
		
		
		for(int i=0;i<hidden_layer-1;i++)
		{
			layers[i]=new Layer();
			layers[i].neurons=new Neuron[no_neurons];

			//setting up the layers with neurons
			if(i==0) // for the first layer
			{	 
				for(int j=0;j<no_neurons;j++)
				{

					layers[i].neurons[j]=new Neuron();
					layers[i].neurons[j].to=new Link[cols];
					//now assigning the incoming links to the neurons
					int k=0;
					while(k< cols)
						{
						   
						   layers[i].neurons[j].to[k++]=links[link_tracker++];
						}
					
				}
			}
			else
			{ //for the neurons in the middle of the network
				for(int j=0;j<no_neurons;j++)
				{

					layers[i].neurons[j]=new Neuron();
					layers[i].neurons[j].to=new Link[cols-1];
					int k=0;
					while(k< cols-1) layers[i].neurons[j].to[k++]=links[link_tracker++];
				}
			}
		}
		layers[hidden_layer-1]=new Layer(); // allocating memory to the last layer
		layers[hidden_layer-1].neurons=new Neuron[1]; //the last layer has a single neuron only
		layers[hidden_layer-1].neurons[0]=new Neuron();
		layers[hidden_layer-1].neurons[0].to=new Link[no_neurons];//allocating links towards the last neuron 
		int k=0;
		while(k< no_neurons) 
			{
				layers[hidden_layer-1].neurons[0].to[k++]=links[link_tracker++];
			}
		
		//so till now we have assigned the weights in the serial order of their index to neurons as and when they are created
		
		int run_count=0;
		int check=0;
		//getting to main business now, will try processing layer by layer
		outer:while(true) //should not run till eternity :D
		{
			run_count++;
			
			for(int r=0;r<rows;r++)
			{
				for(int i=0;i<hidden_layer;i++)
				{
					if(i==0) //the first layer of the network
					{    
						for(int n=0;n<no_neurons;n++)
						{
							double net=0;
							int to_count=0;
							for(int j=0;j<cols-1;j++)
							{
								net+=xor[r][j]*layers[i].neurons[n].to[to_count++].weight;
							}
							net+=1*layers[i].neurons[n].to[to_count++].weight; //here we multiply with the bias
							double op=(double)1/(1+Math.pow(e,-net));        //sigmoidal function as the activation function
							layers[i].neurons[n].output=op;
							
						}
					}
					else if(i==hidden_layer-1) //the last layer of the network,which will have only one neuron
					{
						double net=0; ;int to_count=0;
						for(int n=0;n<no_neurons;n++)
						{
							net+=layers[i-1].neurons[n].output*layers[i].neurons[0].to[to_count++].weight;
						}
						
						double op=(double)1/(1+Math.pow(e,-net)); //sigmoidal function as the activation function
						
						layers[i].neurons[0].output=op;
						System.out.println("output of neuron "+op+" for r:"+r);
					}
					else  //the otherwise case of in between layers
					{
						for(int n=0;n<no_neurons;n++) //for the current layer's neuron
						{
							double net=0;int to_count=0;
							for(int j=0;j<no_neurons;j++)
							{
								net+=layers[i-1].neurons[j].output*layers[i].neurons[j].to[to_count++].weight;
							}
							double op=(double)1/(1+Math.pow(e,-net)); //sigmoidal function as the activation function
							layers[i].neurons[n].output=op;  
						}
					}
				}
				//calculating the error now for back propagation and will update the links
				int t=xor[r][cols-1]; //the expected output
				double o=layers[hidden_layer-1].neurons[0].output;
				
				double delta=scaling_factor*(t-o)*o*(1-o); //weight change at the outer layer
				
				double E=(t-o);
                
				E=Math.pow(E,2);
				E=E/2;
				E=E*100;
				E=Math.round(E);
				E=E/100;
				if(E <= 0.02) //its fine till now since no error
				{
					check++;
					if(check==4)
					{
						System.out.println("We are done with "+run_count+" run(s)");
						break outer;
					}
					else
					{
						continue;
					}
				}
				else
				{
					
					check=0;
					for(int i=hidden_layer-1;i>=0;i--)
					{
						if(i==hidden_layer-1) //the last layer with a single neuron
						{
							layers[i].neurons[0].error=delta;
							for(int n=0;n<no_neurons;n++)
							{
								layers[i].neurons[0].to[n].weight+=scaling_factor*layers[i].neurons[0].error*layers[i-1].neurons[n].output + momentum*layers[i].neurons[0].to[n].delta_weight;
								layers[i].neurons[0].to[n].delta_weight=scaling_factor*layers[i].neurons[0].error;
							}
						}
						
						else if(i==0)  //the first layer
						{
							for(int n=0;n<no_neurons;n++) //neurons for the layer whose incoming weights are to be corrected
							{
								o=layers[i].neurons[n].output;
								for(int j=0;j<layers[i].neurons[n].to.length;j++) //links coming towards the neuron whose weights are to be corrected
								{
									for(k=0;k<layers[i+1].neurons.length;k++)
									{
										//calculating the error from the links going outward and the next layers error
										layers[i].neurons[n].error+=layers[i+1].neurons[k].to[n].weight*layers[i+1].neurons[k].error;
									}
									if(j==layers[i].neurons[n].to.length-1) //this is the -1 case
									{
										layers[i].neurons[n].error*=(1-o)*o;
										layers[i].neurons[n].to[j].weight+=scaling_factor*layers[i].neurons[n].error + momentum*layers[i].neurons[n].to[j].delta_weight; 
										layers[i].neurons[n].to[j].delta_weight=scaling_factor*layers[i].neurons[n].error;
									}
									else
									{	
									  layers[i].neurons[n].error*=(1-o)*o;
									  layers[i].neurons[n].to[j].weight+=scaling_factor*layers[i].neurons[n].error*xor[r][j] + momentum*layers[i].neurons[n].to[j].delta_weight;  
									  layers[i].neurons[n].to[j].delta_weight=scaling_factor*layers[i].neurons[n].error;
									}
								}

							}
						}
						else  //the case otherwise
						{
							//same as above only the output of the input layer has to be changed and one iteration will be less
							for(int n=0;n<no_neurons;n++) //neurons for the layer whose incoming weights are to be corrected
							{
								o=layers[i].neurons[n].output;
								for(int j=0;j<layers[i].neurons[n].to.length;j++) //links coming towards the neuron whose weights are to be corrected
								{
									for(k=0;k<layers[i+1].neurons.length;k++)
									{
										//calculating the error from the links going outward and the next layers error
										layers[i].neurons[n].error+=layers[i+1].neurons[k].to[n].weight*layers[i+1].neurons[k].error;
									}
									  layers[i].neurons[n].error*=(1-o)*o;
									  layers[i].neurons[n].to[j].weight+=scaling_factor*layers[i].neurons[n].error*layers[i-1].neurons[j].output +  momentum*layers[i].neurons[n].to[j].delta_weight;
									  layers[i].neurons[n].to[j].delta_weight=scaling_factor*layers[i].neurons[n].error;  
								}

							}
						}
					}
					r=2; //breaks the outer for loop
				}
			}
		}

	}

}
