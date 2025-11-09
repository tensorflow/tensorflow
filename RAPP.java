public class RAPP
{
    public static void main(String x[])
	{
	   show(5); //initial call 
	}
	public static void show(int x)
	{
	   if(x!=0) //base  case 
	   {  show(x-1);
	   }
	 
	    System.out.println("Good Morning "+x);
		 
	}
}