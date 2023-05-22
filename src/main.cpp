#include <bits/stdc++.h>
#include "include/cnn.h"
#include "read_func.h"
using namespace std;
int main()
{
	vector<case_t> cases = read_train_cases();
	
	Model M;
	M.add_conv( 1, 5, 10, {28, 28, 1} );
	M.add_relu( M.output_size() );
	M.add_pool( 2, 2, M.output_size() );
	M.add_conv( 1, 3, 12, M.output_size() );
	M.add_relu( M.output_size() );
	M.add_pool( 2, 2, M.output_size() );
	M.add_fc( M.output_size(), 10 );
 
	float sum_err = 0;
	int cnt = 0;
	float acc=0;
	cout<<"start training..."<<endl;
	int T=2;
	while(T--)
		for ( case_t& t : cases )
		{
			float xerr = M.train( t.data, t.out );
			sum_err += xerr, cnt++;
			acc += t.out( M.predict(), 0, 0 ) > 0.5 ? 1.0f : 0.0f;
			if ( cnt % 1000 == 0 )
			{
				cout << "cases: " << cnt << " err=" << sum_err/cnt << " acc=" << acc/1000.0 << endl;
				acc=0;
			}
		}
	
	cout << "start testing ..." << endl;
	cases = read_test_cases();
	acc = sum_err = 0;
	for( case_t& t : cases )
	{
		M.forward( t.data ); 
		acc += t.out( M.predict(), 0, 0 ) > 0.5 ? 1.0f : 0.0f;
	}
	cout << "testing results: acc="<< acc/(float)(cases.size()) <<endl;

	return 0;
}
