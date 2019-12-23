# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

file ../../../tools/make/gen/apollo3evb_cortex-m4/bin/pushbutton_cmsis_speech_test
target remote localhost:2331
load ../../../tools/make/gen/apollo3evb_cortex-m4/bin/pushbutton_cmsis_speech_test
monitor reset
break pushbutton_main.c:307
commands
printf "Silence score: %d\n", g_silence_score
printf "Unknown score: %d\n", g_unknown_score
printf "Yes score: %d\n", g_yes_score
printf "No score: %d\n", g_no_score
printf "g_scores[0]: %d\n", g_scores[0]
printf "g_scores[1]: %d\n", g_scores[1]
printf "g_scores[2]: %d\n", g_scores[2]
printf "g_scores[3]: %d\n", g_scores[3]
printf "max_score: %d\n", max_score
printf "max_score_index: %d\n", max_score_index
c
end
c
