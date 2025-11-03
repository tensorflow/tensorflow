"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Navigation from "../components/Navigation";
import type { Appointment } from "../types";

export default function Appointments() {
  const router = useRouter();
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [filter, setFilter] = useState<"all" | "scheduled" | "completed" | "cancelled">("all");
  const [searchTerm, setSearchTerm] = useState("");
  const [showFollowUpModal, setShowFollowUpModal] = useState(false);
  const [selectedAppointment, setSelectedAppointment] = useState<Appointment | null>(null);

  useEffect(() => {
    const stored = localStorage.getItem("appointments");
    if (stored) {
      const parsed = JSON.parse(stored);
      setAppointments(parsed.sort((a: Appointment, b: Appointment) => 
        new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
      ));
    }
  }, []);

  const filteredAppointments = appointments.filter((apt) => {
    const matchesFilter = filter === "all" || apt.status === filter;
    const matchesSearch = 
      apt.patientName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      apt.doctor.toLowerCase().includes(searchTerm.toLowerCase()) ||
      apt.reason.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  const handleCancelAppointment = (id: string) => {
    if (confirm("Are you sure you want to cancel this appointment?")) {
      const updated = appointments.map((apt) =>
        apt.id === id ? { ...apt, status: "cancelled" as const } : apt
      );
      setAppointments(updated);
      localStorage.setItem("appointments", JSON.stringify(updated));
    }
  };

  const handleCompleteAppointment = (id: string) => {
    const updated = appointments.map((apt) =>
      apt.id === id ? { ...apt, status: "completed" as const } : apt
    );
    setAppointments(updated);
    localStorage.setItem("appointments", JSON.stringify(updated));
  };

  const handleScheduleFollowUp = (appointment: Appointment) => {
    setSelectedAppointment(appointment);
    setShowFollowUpModal(true);
  };

  const confirmFollowUp = () => {
    setShowFollowUpModal(false);
    router.push("/book");
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "scheduled":
        return "bg-blue-100 text-blue-800";
      case "completed":
        return "bg-green-100 text-green-800";
      case "cancelled":
        return "bg-red-100 text-red-800";
      default:
        return "bg-gray-100 text-gray-800";
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <Navigation />
      <main className="container mx-auto px-4 py-12">
        <div className="max-w-6xl mx-auto">
          <div className="flex justify-between items-center mb-8">
            <h1 className="text-3xl font-bold text-gray-900">My Appointments</h1>
            <button
              onClick={() => router.push("/book")}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
            >
              Book New Appointment
            </button>
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <div className="flex flex-col md:flex-row gap-4 mb-6">
              <input
                type="text"
                placeholder="Search by patient name, doctor, or reason..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <select
                value={filter}
                onChange={(e) => setFilter(e.target.value as any)}
                className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="all">All Appointments</option>
                <option value="scheduled">Scheduled</option>
                <option value="completed">Completed</option>
                <option value="cancelled">Cancelled</option>
              </select>
            </div>

            {filteredAppointments.length === 0 ? (
              <div className="text-center py-12">
                <div className="text-6xl mb-4">📅</div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  No appointments found
                </h3>
                <p className="text-gray-600 mb-6">
                  {searchTerm || filter !== "all"
                    ? "Try adjusting your search or filter"
                    : "Book your first appointment to get started"}
                </p>
                {!searchTerm && filter === "all" && (
                  <button
                    onClick={() => router.push("/book")}
                    className="bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
                  >
                    Book Appointment
                  </button>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                {filteredAppointments.map((appointment) => (
                  <div
                    key={appointment.id}
                    className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow"
                  >
                    <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-3">
                          <h3 className="text-xl font-semibold text-gray-900">
                            {appointment.patientName}
                          </h3>
                          <span
                            className={`px-3 py-1 rounded-full text-xs font-semibold ${getStatusColor(
                              appointment.status
                            )}`}
                          >
                            {appointment.status.charAt(0).toUpperCase() +
                              appointment.status.slice(1)}
                          </span>
                          {appointment.type === "follow-up" && (
                            <span className="px-3 py-1 rounded-full text-xs font-semibold bg-purple-100 text-purple-800">
                              Follow-up
                            </span>
                          )}
                        </div>

                        <div className="grid md:grid-cols-2 gap-3 text-sm text-gray-600">
                          <div>
                            <span className="font-medium">Date:</span>{" "}
                            {formatDate(appointment.appointmentDate)}
                          </div>
                          <div>
                            <span className="font-medium">Time:</span>{" "}
                            {appointment.appointmentTime}
                          </div>
                          <div>
                            <span className="font-medium">Doctor:</span>{" "}
                            {appointment.doctor}
                          </div>
                          <div>
                            <span className="font-medium">Contact:</span>{" "}
                            {appointment.phone}
                          </div>
                        </div>

                        <div className="mt-3">
                          <span className="font-medium text-sm text-gray-700">
                            Reason:
                          </span>
                          <p className="text-sm text-gray-600 mt-1">
                            {appointment.reason}
                          </p>
                        </div>

                        {appointment.notes && (
                          <div className="mt-3">
                            <span className="font-medium text-sm text-gray-700">
                              Notes:
                            </span>
                            <p className="text-sm text-gray-600 mt-1">
                              {appointment.notes}
                            </p>
                          </div>
                        )}
                      </div>

                      <div className="flex md:flex-col gap-2">
                        {appointment.status === "scheduled" && (
                          <>
                            <button
                              onClick={() => handleCompleteAppointment(appointment.id)}
                              className="px-4 py-2 bg-green-600 text-white rounded-lg text-sm font-semibold hover:bg-green-700 transition-colors"
                            >
                              Complete
                            </button>
                            <button
                              onClick={() => handleScheduleFollowUp(appointment)}
                              className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm font-semibold hover:bg-purple-700 transition-colors"
                            >
                              Follow-up
                            </button>
                            <button
                              onClick={() => handleCancelAppointment(appointment.id)}
                              className="px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-semibold hover:bg-red-700 transition-colors"
                            >
                              Cancel
                            </button>
                          </>
                        )}
                        {appointment.status === "completed" && (
                          <button
                            onClick={() => handleScheduleFollowUp(appointment)}
                            className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm font-semibold hover:bg-purple-700 transition-colors"
                          >
                            Schedule Follow-up
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">
              Appointment Statistics
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-blue-50 rounded-lg">
                <div className="text-3xl font-bold text-blue-600">
                  {appointments.length}
                </div>
                <div className="text-sm text-gray-600 mt-1">Total</div>
              </div>
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <div className="text-3xl font-bold text-green-600">
                  {appointments.filter((a) => a.status === "scheduled").length}
                </div>
                <div className="text-sm text-gray-600 mt-1">Scheduled</div>
              </div>
              <div className="text-center p-4 bg-purple-50 rounded-lg">
                <div className="text-3xl font-bold text-purple-600">
                  {appointments.filter((a) => a.status === "completed").length}
                </div>
                <div className="text-sm text-gray-600 mt-1">Completed</div>
              </div>
              <div className="text-center p-4 bg-red-50 rounded-lg">
                <div className="text-3xl font-bold text-red-600">
                  {appointments.filter((a) => a.status === "cancelled").length}
                </div>
                <div className="text-sm text-gray-600 mt-1">Cancelled</div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {showFollowUpModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg p-8 max-w-md w-full">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Schedule Follow-up
            </h2>
            <p className="text-gray-600 mb-6">
              You will be redirected to the booking page to schedule a follow-up
              appointment for {selectedAppointment?.patientName}.
            </p>
            <div className="flex gap-4">
              <button
                onClick={confirmFollowUp}
                className="flex-1 bg-blue-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
              >
                Continue
              </button>
              <button
                onClick={() => setShowFollowUpModal(false)}
                className="px-6 py-3 border border-gray-300 rounded-lg font-semibold hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
